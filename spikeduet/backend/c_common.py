"""C emission shared by every C backend: literals and the ``model_data.h`` descriptor table."""


from typing import List, Optional


from spikeduet.ir.execution import AffineOp, ExecutionModule, LIFExecutionSpec, LIFUpdateOp


from spikeduet.ir.model import LinearLayer


RESET_MECHANISM_CODE = {"subtract": 0, "zero": 1, "none": 2}


def float_literal(value: float, exact: bool = False) -> str:
    """C float literal. The default (`%.6f`, the legacy convention every frozen header uses) rounds to six
    decimals, which visibly truncates SMALL weights; `exact=True` prints 9 significant digits, which
    round-trips every float32 losslessly."""
    if not exact:
        return f"{value:.6f}f"
    text = f"{float(value):.9g}"
    if not any(c in text for c in ".eEn"):          # '1' -> '1.0' (a bare '1f' is not a C literal)
        text += ".0"
    return text + "f"


def array_literal(values, exact: bool = False) -> str:
    return "{ " + ", ".join(float_literal(v, exact) for v in values) + " }"


def quantize_rows_int8(weight):
    """Per-output-row symmetric int8 quantization of a `[out, in]` weight matrix: `scale[j] = max|w[j]| / 127`
    (float32; an all-zero row gets scale 1.0), `q = round-half-even(w / scale)` clipped to +-127. The device
    reconstructs `w ~ q * scale[j]` in float32. Returns `(q int8 [out, in], scale float32 [out])`."""
    import numpy as np

    w = np.asarray(weight, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def layer_sequence(module: ExecutionModule):
    """Returns ``(linear_layers, lif_specs)`` in program order from the Execution IR."""
    linear_layers: List[LinearLayer] = []
    lif_specs: List[LIFExecutionSpec] = []
    for op in module.walk_ops():
        if isinstance(op, AffineOp):
            linear_layers.append(op.weight_layer)
        elif isinstance(op, LIFUpdateOp):
            lif_specs.append(op.semantics)
    return linear_layers, lif_specs


def extension_layer_sequence(module: ExecutionModule):
    """FilterBank-extension view of the layers: per layer, the
    `(delays_tensor_or_None, LIFExecutionSpec)` pair. Used to decide whether
    the optional extension tables are emitted at all."""
    linear_layers, lif_specs = layer_sequence(module)
    return [(l.synapse_delays, s) for l, s in zip(linear_layers, lif_specs)]


def module_uses_extensions(module: ExecutionModule) -> bool:
    """True iff the module uses any FilterBank extension (per-synapse
    delays, spike-state / adaptive-threshold LIF, weighted readout). When
    False, `render_model_data_header` emits NO extension text at all, so
    every pre-extension model's generated header is byte-identical to before."""
    if module.readout_weights is not None:
        return True
    return any(d is not None or s.spike_state or s.adaptation is not None for d, s in extension_layer_sequence(module))


def _render_extension_tables(module: ExecutionModule, exact: bool = False) -> List[str]:
    """Optional `iscas_layer_ext_t ISCAS_LAYER_EXT[]` (parallel to
    `ISCAS_LAYERS[]`) + `ISCAS_READOUT_WEIGHTS[]`. Kept in separate tables
    rather than extra `iscas_layer_t` fields so the base descriptor (and every
    header generated without these features) is unchanged."""
    lines = [
        "",
        "/* ---- FilterBank extension (per-synapse delays, spike-state / adaptive-threshold LIF,",
        " * weighted readout). Present only for models that use them. ---- */",
        "#define ISCAS_HAS_LAYER_EXT 1",
        "typedef struct iscas_layer_ext_t {",
        "    const unsigned char *synapse_delays; /* [out_features][in_features] delay in timesteps, or NULL (no delays) */",
        "    int delay_depth;                      /* max delay D: the input-history ring holds D+1 slots; 0 if no delays */",
        "    int spike_state;                      /* 1: the delayed reset is keyed on a stored previous spike */",
        "    const float *alif_rho;                /* adaptation decay [alif_rho_len] (1 or per-neuron), NULL if none */",
        "    int alif_rho_len;                     /* 0: no adaptive threshold */",
        "    const float *alif_a;                  /* adaptation gain [alif_a_len] */",
        "    int alif_a_len;",
        "} iscas_layer_ext_t;",
        "",
    ]
    seq = extension_layer_sequence(module)
    for i, (delays, spec) in enumerate(seq):
        if delays is not None:
            flat = [int(v) for v in delays.reshape(-1).tolist()]
            if max(flat) > 255 or min(flat) < 0:
                raise ValueError(f"layer {i} synapse delays must lie in [0, 255]")
            lines.append(
                f"static const unsigned char iscas_layer{i}_synapse_delays[{len(flat)}] = {{ "
                + ", ".join(str(v) for v in flat) + " };"
            )
        if spec.adaptation is not None:
            lines.append(f"static const float iscas_layer{i}_alif_rho[{len(spec.adaptation.rho)}] = {array_literal(spec.adaptation.rho, exact)};")
            lines.append(f"static const float iscas_layer{i}_alif_a[{len(spec.adaptation.a)}] = {array_literal(spec.adaptation.a, exact)};")
    lines.append("")
    lines.append("static const iscas_layer_ext_t ISCAS_LAYER_EXT[ISCAS_NUM_LAYERS] = {")
    for i, (delays, spec) in enumerate(seq):
        d_ref = f"iscas_layer{i}_synapse_delays" if delays is not None else "NULL"
        depth = int(delays.max()) if delays is not None else 0
        if spec.adaptation is not None:
            a = spec.adaptation
            alif = f"iscas_layer{i}_alif_rho, {len(a.rho)}, iscas_layer{i}_alif_a, {len(a.a)}"
        else:
            alif = "NULL, 0, NULL, 0"
        lines.append(f"    {{ {d_ref}, {depth}, {1 if spec.spike_state else 0}, {alif} }},")
    lines.append("};")
    if module.readout_weights is not None:
        lines += [
            "",
            f"#define ISCAS_READOUT_WEIGHTED 1",
            f"#define ISCAS_READOUT_WEIGHTS_LEN {len(module.readout_weights)}",
            "/* readout = argmax of sum_t ISCAS_READOUT_WEIGHTS[t] * (final-layer membrane at step t) */",
            f"static const float ISCAS_READOUT_WEIGHTS[{len(module.readout_weights)}] = {array_literal(module.readout_weights, exact)};",
        ]
    return lines


def render_model_data_header(
    module: ExecutionModule,
    *,
    guard: str = "MODEL_DATA_H",
    generator_comment: str = "// Generated by SpikeDuet -- do not edit by hand",
    exact_floats: bool = False,
    int8_weights: bool = False,
) -> str:
    """Renders the `iscas_layer_t` descriptor table + weight/bias/beta/
    threshold/graded_spikes_factor/state_quant arrays for `module`, as a
    self-contained C header. Vendor-neutral: only `<stddef.h>` and plain
    `static const` array declarations -- safe to `#include` from either
    the EFR32 firmware tree or a portable-C build with no vendor headers
    at all.

    `guard` lets two backends emit this header under different
    `#ifndef`/`#define` names (so, in principle, both could be included
    in the same translation unit without collision) -- content is
    otherwise identical for the same `module`.

    `int8_weights` (opt-in; the EFR32 dense-MVP engine only): each layer's weights are stored as an `int8_t`
    table + one float scale per output row (`quantize_rows_int8`) in `ISCAS_QWEIGHTS[]` instead of an FP32
    array, and `iscas_layer_t.weight` is NULL. The engine rebuilds its FP16 weight copy from them in `snn_init()`.
    """
    linear_layers, lif_specs = layer_sequence(module)

    input_dim = linear_layers[0].in_features
    output_dim = linear_layers[-1].out_features
    num_layers = len(linear_layers)

    lines = [
        generator_comment,
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        *(["#include <stdint.h>"] if int8_weights else []),
        "",
        f"#define ISCAS_INPUT_DIM {input_dim}",
        f"#define ISCAS_OUTPUT_DIM {output_dim}",
        f"#define ISCAS_NUM_LAYERS {num_layers}",
        "",
        "typedef struct iscas_layer_t {",
        "    int in_features;",
        "    int out_features;",
        "    const float *weight;  /* row-major [out_features][in_features], flattened */",
        "    const float *bias;    /* [out_features], or NULL if the layer has no bias */",
        "    const float *beta;      /* [beta_len]: beta_len==1 -> shared scalar, beta_len==out_features -> per-neuron */",
        "    int beta_len;",
        "    const float *threshold; /* [threshold_len]: same convention as beta */",
        "    int threshold_len;",
        "    const float *graded_spikes_factor; /* [graded_spikes_factor_len]: same convention as beta; scales only the EMITTED spike, never reset math */",
        "    int graded_spikes_factor_len;",
        "    int reset_mechanism;  /* 0=subtract, 1=zero, 2=none -- snntorch.SpikingNeuron.reset_dict numbering */",
        "    int reset_delay;      /* 0=immediate (same-timestep) reset, 1=deferred one timestep (snnTorch's own default) */",
        "    int inhibition;       /* 0=independent per-neuron update, 1=layer-wide winner-take-all */",
        "    const float *state_quant_levels; /* [state_quant_num_levels], NULL if disabled; always per-layer, never per-neuron */",
        "    int state_quant_num_levels;      /* 0 if disabled */",
        "} iscas_layer_t;",
        "",
        *(["typedef struct iscas_qweight_t {",
           "    const int8_t *q;      /* [out_features][in_features], row-major */",
           "    const float *scale;   /* [out_features]: weight ~ q * scale[row] */",
           "} iscas_qweight_t;",
           ""] if int8_weights else []),
    ]

    bias_refs: List[Optional[str]] = []
    for i, linear in enumerate(linear_layers):
        weight_name = f"iscas_layer{i}_weight"
        if int8_weights:
            q, scale = quantize_rows_int8(linear.weight)
            lines.append(f"static const int8_t {weight_name}_q[{q.size}] = {{ " + ", ".join(str(v) for v in q.reshape(-1).tolist()) + " };")
            lines.append(f"static const float {weight_name}_scale[{scale.size}] = {array_literal(scale.tolist(), True)};")
        else:
            flat_weight = linear.weight.reshape(-1).tolist()
            lines.append(f"static const float {weight_name}[{len(flat_weight)}] = {array_literal(flat_weight, exact_floats)};")
        if linear.bias is not None:
            bias_name = f"iscas_layer{i}_bias"
            flat_bias = linear.bias.reshape(-1).tolist()
            lines.append(f"static const float {bias_name}[{len(flat_bias)}] = {array_literal(flat_bias, exact_floats)};")
            bias_refs.append(bias_name)
        else:
            bias_refs.append(None)

        lif = lif_specs[i]
        beta_name = f"iscas_layer{i}_beta"
        threshold_name = f"iscas_layer{i}_threshold"
        lines.append(f"static const float {beta_name}[{len(lif.beta)}] = {array_literal(lif.beta, exact_floats)};")
        lines.append(f"static const float {threshold_name}[{len(lif.threshold)}] = {array_literal(lif.threshold, exact_floats)};")
        graded_name = f"iscas_layer{i}_graded_spikes_factor"
        lines.append(
            f"static const float {graded_name}[{len(lif.graded_spikes_factor)}] = "
            f"{array_literal(lif.graded_spikes_factor, exact_floats)};"
        )
        if lif.state_quant is not None:
            levels_name = f"iscas_layer{i}_state_quant_levels"
            lines.append(
                f"static const float {levels_name}[{len(lif.state_quant.levels)}] = "
                f"{array_literal(lif.state_quant.levels, exact_floats)};"
            )
        lines.append("")

    lines.append("static const iscas_layer_t ISCAS_LAYERS[ISCAS_NUM_LAYERS] = {")
    for i, (linear, lif) in enumerate(zip(linear_layers, lif_specs)):
        bias_ref = bias_refs[i] if bias_refs[i] is not None else "NULL"
        reset_code = RESET_MECHANISM_CODE[lif.reset_mechanism]
        reset_delay_code = 1 if lif.reset_delay else 0
        inhibition_code = 1 if lif.inhibition else 0
        if lif.state_quant is not None:
            state_quant_ref = f"iscas_layer{i}_state_quant_levels"
            state_quant_len = len(lif.state_quant.levels)
        else:
            state_quant_ref = "NULL"
            state_quant_len = 0
        lines.append(
            f"    {{ {linear.in_features}, {linear.out_features}, "
            f"{'NULL' if int8_weights else f'iscas_layer{i}_weight'}, {bias_ref}, "
            f"iscas_layer{i}_beta, {len(lif.beta)}, "
            f"iscas_layer{i}_threshold, {len(lif.threshold)}, "
            f"iscas_layer{i}_graded_spikes_factor, {len(lif.graded_spikes_factor)}, "
            f"{reset_code}, {reset_delay_code}, {inhibition_code}, "
            f"{state_quant_ref}, {state_quant_len} }},"
        )
    lines.append("};")
    if int8_weights:
        lines.append("static const iscas_qweight_t ISCAS_QWEIGHTS[ISCAS_NUM_LAYERS] = {")
        lines.extend(f"    {{ iscas_layer{i}_weight_q, iscas_layer{i}_weight_scale }}," for i in range(num_layers))
        lines.append("};")
    if module_uses_extensions(module):
        lines.extend(_render_extension_tables(module, exact_floats))
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")

    return "\n".join(lines)
