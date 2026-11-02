"""Op-by-op C emission of an Execution IR schedule, shared by the portable-C and EFR32 backends.

Every op becomes its own unrolled C block, in program order: preheader ops go before the
timestep loop, loop-body ops inside it. Placement is therefore structural (there is no
``if (t == 0)`` anywhere). Only the affine realization differs between backends, so it is passed
in as an ``AffineEmitter``; the LIF emitter can be replaced the same way. LIF semantics are never
implemented here: the emitted code calls the backend's ``iscas_lif_update()``,
``iscas_lif_update_inhibition_layer()`` and ``iscas_param_at()``.

Storage convention: ``g_stage``, ``g_current``, ``g_mem``, ``g_spike``, ``g_inhib_mem_state`` and
``g_inhib_reset_old``, each indexed ``[layer][neuron]``.
"""


import re


from dataclasses import dataclass


from typing import FrozenSet, Callable, List, Optional, Tuple


from spikeduet.ir.execution import AffineOp, ConvertOp, DelayPushOp, ExecutionModule, LIFUpdateOp, op_outputs


from spikeduet.ir.execution_analysis import consumers, producers


AffineEmitter = Callable[[AffineOp, int], List[str]]


LifEmitter = Callable[[LIFUpdateOp], List[str]]


def layer_index_of(buffer_name: str) -> int:
    """Trailing layer index of a lowered buffer name (``"current2"`` -> ``2``). Passes move ops
    but never rename buffers, so this index is also the layer's position in ``ISCAS_LAYERS``."""
    m = re.search(r"(\d+)$", buffer_name)
    if not m:
        raise ValueError(f"buffer name {buffer_name!r} has no trailing layer index")
    return int(m.group(1))


_ARRAY_PREFIXES = {
    "stage": "g_stage",
    "current": "g_current",
    "mem": "g_mem",
    "spike": "g_spike",
    # FilterBank extension buffers (only ever present for models that use them)
    "hist": "g_hist",
    "adapt": "g_adapt",
    "sprev": "g_sprev",
}


@dataclass(frozen=True)
class ExtensionUsage:
    """Which FilterBank extensions an Execution IR module uses (all False for
    every ordinary model, in which case backends emit no extension text and
    their output is byte-identical to before the extensions existed)."""

    delays: bool
    spike_state: bool
    adaptation: bool
    weighted_readout: bool

    @property
    def any(self) -> bool:
        return self.delays or self.spike_state or self.adaptation or self.weighted_readout

    def describe(self) -> str:
        return ", ".join(n for n, f in (("per-synapse delays", self.delays), ("spike-state LIF", self.spike_state),
                                          ("adaptive threshold", self.adaptation),
                                          ("weighted readout", self.weighted_readout)) if f)


def extension_usage(module: ExecutionModule) -> ExtensionUsage:
    ops = module.walk_ops()
    return ExtensionUsage(
        delays=any(isinstance(o, DelayPushOp) for o in ops),
        spike_state=any(isinstance(o, LIFUpdateOp) and o.spike_state is not None for o in ops),
        adaptation=any(isinstance(o, LIFUpdateOp) and o.adapt is not None for o in ops),
        weighted_readout=module.readout_weights is not None,
    )


def delay_bit_layers(module: ExecutionModule) -> dict:
    """`{layer_index: max_delay}` for every delayed layer whose input history can be kept as ONE BITMASK
    WORD per input neuron instead of a float ring.

    Legal iff the delay line is fed (through its staging `ConvertOp`) by a `LIFUpdateOp` whose emitted
    spikes are provably binary -- unit `graded_spikes_factor` everywhere (inhibition emits binary spikes
    too) -- and the maximum delay fits a 64-bit word. Then bit `d` of `hbits[i]` is "input neuron i spiked
    d steps ago". Pushing is one shift-or per input neuron (no memmove of the whole ring) and the delayed
    gather can skip every synapse whose delayed spike is 0. Skipped terms are exact zeros
    (`w * 0.0f`), added in the same ascending-input order, so the layer's currents are bit-identical to the
    float-ring realization. Anything else (raw external input, graded spikes, delays > 63) keeps the
    float ring."""
    prod = producers(module)
    ops = {o.id: o for o in module.walk_ops()}
    out = {}
    for op in module.walk_ops():
        if not isinstance(op, DelayPushOp):
            continue
        src = ops.get(prod.get(op.input))
        if isinstance(src, ConvertOp):
            src = ops.get(prod.get(src.input))
        if binary_spike_buffer(module, op.input) and op.depth <= 63:
            out[layer_index_of(op.history)] = op.depth
    return out


def binary_spike_buffer(module: ExecutionModule, buffer_name: str) -> bool:
    """True iff `buffer_name` (a staging buffer, or a LIF's spike buffer) provably holds only 0.0f / 1.0f:
    it is written by a `LIFUpdateOp` with unit graded_spikes_factor, directly or through one staging
    `ConvertOp`. (Inhibition emits unscaled binary spikes as well.)"""
    prod = producers(module)
    ops = {o.id: o for o in module.walk_ops()}
    src = ops.get(prod.get(buffer_name))
    if isinstance(src, ConvertOp):
        src = ops.get(prod.get(src.input))
    return isinstance(src, LIFUpdateOp) and all(g == 1.0 for g in src.semantics.graded_spikes_factor)


def delay_bit_typedef(bits: dict) -> str:
    """C type of the per-input bitmask word for the given `delay_bit_layers` result ('' if none)."""
    if not bits:
        return ""
    return "typedef uint32_t iscas_hbits_t;" if max(bits.values()) <= 31 else "typedef uint64_t iscas_hbits_t;"


def reject_extensions(module: ExecutionModule, target: str) -> None:
    """Loudly refuse to generate `target` code for a module that uses a
    FilterBank extension this realization does not implement (never a silent
    approximation)."""
    use = extension_usage(module)
    if use.any:
        raise ValueError(
            f"{target} does not support the FilterBank extensions used by this model "
            f"({use.describe()}); use the scalar CPU engine or the portable-C backend"
        )


def buffer_ref(buffer_name: str) -> str:
    """C storage expression of a buffer: ``"stage1"`` -> ``"g_stage[1]"``."""
    for prefix, array in _ARRAY_PREFIXES.items():
        if buffer_name.startswith(prefix):
            return f"{array}[{layer_index_of(buffer_name)}]"
    raise ValueError(f"buffer name {buffer_name!r} has no known storage array prefix")


def emit_convert(
    op: ConvertOp,
    module: ExecutionModule,
    *,
    in_preheader: bool,
    input_param: str,
    input_temporal_macro: str,
    input_dim_macro: str,
    direct_input_source: bool = False,
) -> List[str]:
    """A staging-copy block for one ``ConvertOp``.

    When the op reads the external input: in the preheader (invariant input) it reads
    ``input_param`` directly; in the loop it selects step ``t`` when ``input_temporal_macro`` is
    set. With ``direct_input_source`` the caller already passes the current step's vector, so no
    offset is computed (used by the EFR32 streaming step function). Otherwise it copies the
    previous layer's spikes.
    """
    idx = layer_index_of(op.output)
    if op.input == module.input_buffer:
        if in_preheader or direct_input_source:
            source = input_param
        else:
            source = (
                f"({input_temporal_macro} ? "
                f"({input_param} + (size_t)t * {input_dim_macro}) : {input_param})"
            )
    else:
        source = buffer_ref(op.input)
    return [
        f"    {{ /* {op.id}: convert -> {op.output} */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        const float *source = {source};",
        f"        for (int i = 0; i < L->in_features; i++) {{ {buffer_ref(op.output)}[i] = source[i]; }}",
        f"        g_convert_exec_count[{idx}]++;",
        "    }",
    ]


def scalar_affine_emitter(op: AffineOp, idx: int) -> List[str]:
    """Plain scalar-C `AffineOp` realization -- shared by portable-C and
    EFR32-CPU (identical arithmetic, identical operand order: bias-then-
    accumulate, matching `firmware/common/snn_generic_cpu.c`)."""
    stage_ref = buffer_ref(op.input)
    if op.weight_layer.synapse_delays is not None:
        # per-synapse delays: `stage_ref` is the history ring, hist[d * in + i] = input i, d steps ago
        return [
            f"    {{ /* {op.id}: affine layer {idx} (scalar, per-synapse delays) */",
            f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
            f"        const iscas_layer_ext_t *E = &ISCAS_LAYER_EXT[{idx}];",
            "        for (int j = 0; j < L->out_features; j++) {",
            "            float c = L->bias ? L->bias[j] : 0.0f;",
            "            for (int i = 0; i < L->in_features; i++) {",
            "                const int k = j * L->in_features + i;",
            f"                c += L->weight[k] * {stage_ref}[(size_t)E->synapse_delays[k] * L->in_features + i];",
            "            }",
            f"            g_current[{idx}][j] = c;",
            "        }",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
        ]
    return [
        f"    {{ /* {op.id}: affine layer {idx} (scalar) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        "        for (int j = 0; j < L->out_features; j++) {",
        "            float c = L->bias ? L->bias[j] : 0.0f;",
        "            for (int i = 0; i < L->in_features; i++) {",
        f"                c += L->weight[(size_t)j * L->in_features + i] * {stage_ref}[i];",
        "            }",
        f"            g_current[{idx}][j] = c;",
        "        }",
        f"        g_affine_exec_count[{idx}]++;",
        "    }",
    ]


def mvp_affine_emitter(op: AffineOp, idx: int) -> List[str]:
    """MVP-accelerated `AffineOp` realization -- EFR32-MVP only. FP16
    input staging + one `sl_math_mvp_matrix_vector_mult_f16()` call +
    float32 bias add, matching `firmware/common/snn_generic_mvp.c`'s
    existing per-layer pattern (weight-to-FP16 conversion and
    `sl_math_matrix_init_f16` happen once at `snn_init()`, generated
    separately -- see `compiler/backend/efr32.py`).

    Executed exactly once if `idx`'s `AffineOp` was hoisted into the
    preheader, `ISCAS_STEPS` times otherwise -- the MVP call SITE itself
    is what's placed inside or outside the loop by the caller (this
    function just emits the block; `emit_schedule` decides where it
    goes), never a runtime skip predicate around an always-executed call.

    B4 Phase B2 note: `SpikeStagingFusionPass` eliminates the redundant
    FP32 `ConvertOp` copy feeding `op.input` (a fused upstream `LIFUpdateOp`
    now writes this layer's `stage{idx}` buffer directly) -- but through
    Phase B2, the FP32->FP16 cast into `g_in_f16[idx]` below still always
    ran, unchanged, regardless of fusion.

    B4 Phase B2.5: `compiler/backend/efr32.py`'s MVP generator now calls
    THIS function only for edges that are NOT compiler-proven direct-FP16
    candidates (see its `_mvp_direct_fused_layer_indices`/
    `_mvp_affine_emitter` -- a fused, non-inhibited edge instead gets a
    staging-loop-free variant, since the upstream `LIFUpdateOp` already
    wrote `g_in_f16[idx]` directly). This function -- the ORIGINAL staged
    path -- remains exactly as before (still used for: unfused edges,
    `baseline_pipeline()`/`licm_pipeline()` builds where fusion never ran,
    and inhibited fused edges, which Phase B2.5 deliberately keeps on this
    staged path -- see that module's docstring for why). The FP16
    conversion itself is NEVER silently dropped on this path -- only the
    redundant FP32 identity copy was, back in Phase B2.
    """
    if op.weight_layer.synapse_delays is not None:
        raise ValueError("the MVP affine realization does not support per-synapse delays (use the scalar CPU engine)")
    stage_ref = buffer_ref(op.input)
    return [
        f"    {{ /* {op.id}: affine layer {idx} (MVP, staged FP16 input) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        for (int i = 0; i < L->in_features; i++) {{ g_in_f16[{idx}][i] = (float16_t){stage_ref}[i]; }}",
        f"        g_mvp_stage_cast_count[{idx}]++;",
        f"        sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_cur_f16[{idx}]);",
        "        for (int j = 0; j < L->out_features; j++) {",
        f"            g_current[{idx}][j] = (float)g_cur_f16[{idx}][j] + (L->bias ? L->bias[j] : 0.0f);",
        "        }",
        f"        g_affine_exec_count[{idx}]++;",
        "    }",
    ]


def emit_delay_push(op: DelayPushOp) -> List[str]:
    """One `DelayPushOp` -> shift the history ring one slot older and write the
    staged input into slot 0 (slot d = d timesteps ago)."""
    idx = layer_index_of(op.history)
    return [
        f"    {{ /* {op.id}: delay-line push {op.input} -> {op.history} (depth {op.depth}) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        float *h = {buffer_ref(op.history)};",
        f"        memmove(h + L->in_features, h, (size_t){op.depth} * (size_t)L->in_features * sizeof(float));",
        f"        for (int i = 0; i < L->in_features; i++) {{ h[i] = {buffer_ref(op.input)}[i]; }}",
        "    }",
    ]


def emit_delay_push_bits(op: DelayPushOp, source: Optional[str] = None) -> List[str]:
    """Bitmask realization of a `DelayPushOp` (see `delay_bit_layers`): bit d of hbits[i] = input i spiked
    d steps ago. One shift-or per input; the staged input is exactly 0.0f or 1.0f."""
    idx = layer_index_of(op.history)
    return [
        f"    {{ /* {op.id}: delay-line push {op.input} -> bitmask history (depth {op.depth}) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        iscas_hbits_t *h = g_hbits[{idx}];",
        f"        for (int i = 0; i < L->in_features; i++) {{ h[i] = (iscas_hbits_t)((h[i] << 1) | (iscas_hbits_t)({source or buffer_ref(op.input)}[i] != 0.0f)); }}",
        "    }",
    ]


UNROLL_DELAY_SYNAPSE_LIMIT = 1024


def emit_affine_spike_gated(op: AffineOp, idx: int) -> List[str]:
    """Spike-gated plain affine for a provably binary input (`binary_spike_buffer`): bias, then for each
    input that is 1 the weight column is added, in ascending input order. `w * 1.0f` is `w` and `w * 0.0f` is
    an exact zero, so the result is bit-identical to `scalar_affine_emitter`'s multiply-accumulate."""
    stage_ref = buffer_ref(op.input)
    return [
        f"    {{ /* {op.id}: affine layer {idx} (scalar, spike-gated: binary input) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        const float *x = {stage_ref};",
        f"        float *cur = g_current[{idx}];",
        "        for (int j = 0; j < L->out_features; j++) {",
        "            cur[j] = L->bias ? L->bias[j] : 0.0f;",
        "        }",
        "        for (int i = 0; i < L->in_features; i++) {",
        "            if (x[i] != 0.0f) {",
        "                for (int j = 0; j < L->out_features; j++) { cur[j] += L->weight[(size_t)j * L->in_features + i]; }",
        "            }",
        "        }",
        f"        g_affine_exec_count[{idx}]++;",
        "    }",
    ]


def emit_affine_delay_bits(op: AffineOp, idx: int, depth: int) -> List[str]:
    """Spike-gated delayed affine over the bitmask history (see `delay_bit_layers`). For each output j:
    bias, then the weights of the synapses whose delayed input spike is 1, added in ascending input order --
    the same sequence of non-zero additions the float-ring realization performs, so the result is
    bit-identical (the skipped terms are `w * 0.0f`, i.e. exact zeros; a weight that prints as exactly
    0.000000f is skipped too). Input neurons with no spike in the whole window are skipped outright.

    Up to `UNROLL_DELAY_SYNAPSE_LIMIT` synapses the code is fully unrolled with the delays and weights as
    literals (weights printed with the same 6-decimal convention as `model_data.h`, so the values are
    bit-for-bit the descriptor table's); beyond that a generic loop reads them from the descriptor."""
    layer = op.weight_layer
    head = [
        f"    {{ /* {op.id}: affine layer {idx} (scalar, per-synapse delays, spike-gated bitmask history) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        const iscas_hbits_t *h = g_hbits[{idx}];",
        f"        const iscas_hbits_t mask = (iscas_hbits_t){hex((1 << (depth + 1)) - 1)}ULL;",
        f"        float *cur = g_current[{idx}];",
        "        for (int j = 0; j < L->out_features; j++) {",
        "            cur[j] = L->bias ? L->bias[j] : 0.0f;",
        "        }",
    ]
    tail = [f"        g_affine_exec_count[{idx}]++;", "    }"]
    if layer.out_features * layer.in_features > UNROLL_DELAY_SYNAPSE_LIMIT:
        return head + [
            f"        const iscas_layer_ext_t *E = &ISCAS_LAYER_EXT[{idx}];",
            "        for (int i = 0; i < L->in_features; i++) {",
            "            const iscas_hbits_t bits = h[i] & mask;",
            "            if (bits == 0) { continue; }",
            "            for (int j = 0; j < L->out_features; j++) {",
            "                const int k = j * L->in_features + i;",
            "                if ((bits >> E->synapse_delays[k]) & 1u) { cur[j] += L->weight[k]; }",
            "            }",
            "        }",
        ] + tail
    delays, weight = layer.synapse_delays.tolist(), layer.weight.tolist()
    body = []
    for i in range(layer.in_features):
        taps = []
        for j in range(layer.out_features):
            lit = f"{weight[j][i]:.6f}f"
            if float(lit[:-1]) == 0.0:
                continue
            taps.append(f"            if (b & ((iscas_hbits_t)1 << {delays[j][i]})) {{ cur[{j}] += {lit}; }}")
        if taps:
            body += [f"        {{ const iscas_hbits_t b = h[{i}] & mask;   /* input {i} */", "          if (b) {", *taps, "          }", "        }"]
    return head + body + tail


def emit_lif_spike_state(op: LIFUpdateOp) -> List[str]:
    """A `LIFUpdateOp` whose delayed reset is keyed on the stored previous spike
    (optionally with an adaptive threshold) -> a per-neuron loop over the shared
    `iscas_lif_update_spike_state()` helper. Same buffer-resolution rules as
    `emit_lif` (the write target is resolved from the buffer's own name, so a
    fused op needs no special case)."""
    idx = layer_index_of(op.current)
    current_ref, mem_ref, spike_ref = buffer_ref(op.current), buffer_ref(op.membrane), buffer_ref(op.output_spike)
    sprev_ref = buffer_ref(op.spike_state)
    has_adapt = op.adapt is not None
    lines = [
        f"    {{ /* {op.id}: lif layer {idx} (stored previous spike"
        + (", adaptive threshold" if has_adapt else "") + f") -> {op.output_spike} */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
    ]
    if has_adapt:
        lines.append(f"        const iscas_layer_ext_t *E = &ISCAS_LAYER_EXT[{idx}];")
    lines += [
        "        for (int j = 0; j < L->out_features; j++) {",
        "            float spike, new_mem;",
    ]
    if has_adapt:
        adapt_ref = buffer_ref(op.adapt)
        lines += [
            f"            float *adapt = &{adapt_ref}[j];",
            "            const float rho = iscas_param_at(E->alif_rho, E->alif_rho_len, j);",
            "            const float gain = iscas_param_at(E->alif_a, E->alif_a_len, j);",
        ]
        adapt_args, ha = "rho, gain, adapt", 1
    else:
        lines.append("            float unused_adapt = 0.0f;")
        adapt_args, ha = "0.0f, 0.0f, &unused_adapt", 0
    lines += [
        f"            iscas_lif_update_spike_state({current_ref}[j], {mem_ref}[j],",
        "                iscas_param_at(L->beta, L->beta_len, j),",
        "                iscas_param_at(L->threshold, L->threshold_len, j),",
        f"                {sprev_ref}[j], {ha}, {adapt_args}, &spike, &new_mem);",
        f"            {mem_ref}[j] = new_mem;",
        f"            {sprev_ref}[j] = spike;",
        f"            {spike_ref}[j] = spike;",
        "        }",
        f"        g_lif_exec_count[{idx}]++;",
        "    }",
    ]
    return lines


def plain_delayed_subtract(sem) -> bool:
    """A LIF whose recurrence is exactly `new_mem = beta*old + cur - (old > th)*th; spike = new_mem > th`:
    subtract reset, delayed one step, no inhibition / state quantization / graded scaling / adaptation /
    stored-spike state. (snnTorch's own default `Leaky` configuration.)"""
    return (sem.reset_mechanism == "subtract" and sem.reset_delay and not sem.inhibition and sem.state_quant is None
            and all(g == 1.0 for g in sem.graded_spikes_factor) and sem.adaptation is None and not sem.spike_state)


def emit_lif_fast(op: LIFUpdateOp, fp16_target: Optional[int] = None, active: Optional[Tuple[str, str]] = None) -> List[str]:
    """Specialised LIF loop for `plain_delayed_subtract` layers (opt-in via `speed`): the layer sizes and the
    scalar-vs-per-neuron parameter choice are resolved at generation time, so the loop body is just the
    recurrence -- no reset-mechanism / quantization / graded-factor branches. The arithmetic expression is
    the one `iscas_lif_update()` evaluates for this configuration, so results are bit-identical.
    `fp16_target` set: the spike goes straight into the MVP FP16 input buffer `g_in_f16[fp16_target]`
    (EFR32-MVP direct-FP16 edge), otherwise into `op.output_spike`. `active=(array, count)` restricts the loop
    to the listed neurons (see `silent_lif_plans`)."""
    sem = op.semantics
    idx = layer_index_of(op.current)
    n = sem.num_neurons
    current_ref, mem_ref = buffer_ref(op.current), buffer_ref(op.membrane)
    scalar_beta, scalar_th = len(sem.beta) == 1, len(sem.threshold) == 1
    dest = f"g_in_f16[{fp16_target}][j] = (float16_t)spike" if fp16_target is not None else f"{buffer_ref(op.output_spike)}[j] = spike"
    where = f"direct FP16 g_in_f16[{fp16_target}] (MVP-fused)" if fp16_target is not None else op.output_spike
    lines = [
        f"    {{ /* {op.id}: lif layer {idx} -> {where} (specialised: subtract, delayed reset) */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        f"        const float *cur = {current_ref};",
        f"        float *mem = {mem_ref};",
    ]
    if scalar_beta:
        lines.append("        const float beta0 = L->beta[0];")
    if scalar_th:
        lines.append("        const float th0 = L->threshold[0];")
    if active is None:
        loop_head = [f"        for (int j = 0; j < {n}; j++) {{"]
    else:
        loop_head = [f"        for (int a = 0; a < {active[1]}; a++) {{", f"            const int j = {active[0]}[a];"]
    lines += [
        *loop_head,
        f"            const float beta = {'beta0' if scalar_beta else 'L->beta[j]'};",
        f"            const float th = {'th0' if scalar_th else 'L->threshold[j]'};",
        "            const float old_mem = mem[j];",
        "            const float reset_old = (old_mem > th) ? 1.0f : 0.0f;",
        "            const float new_mem = (beta * old_mem) + cur[j] - reset_old * th;",
        "            const float spike = (new_mem > th) ? 1.0f : 0.0f;",
        "            mem[j] = new_mem;",
        f"            {dest};",
        "        }",
        f"        g_lif_exec_count[{idx}]++;",
    ]
    if fp16_target is not None:
        lines.append(f"        g_mvp_direct_fp16_write_count[{fp16_target}]++;")
    lines.append("    }")
    return lines


@dataclass(frozen=True)
class SilentLifPlan:
    """Silent-neuron elimination for a LIF driven by a time-INVARIANT current (see `silent_lif_plans`)."""

    lif_id: str
    idx: int
    n: int
    current: str
    fp16_target: Optional[int]


def silent_lif_plans(module: ExecutionModule, direct_fused=frozenset()) -> dict:
    """`{lif op id: SilentLifPlan}` for every LIF whose per-neuron current is computed ONCE in the preheader
    (static input + hoisted affine), so its whole spike train is fixed by that constant current.

    With `v_1 = cur`, `v_(t+1) = beta*v_t + cur` (no reset until the first spike), `v_t < cur / (1 - beta)`:
    a neuron with `cur <= 0.999 * threshold * (1 - beta)` can never reach its threshold (the 0.1% margin
    dwarfs float rounding for the supported run lengths) and emits 0 on every step. The plan builds the list
    of the OTHER neurons once, zeroes the silent outputs once, and the per-step loop visits only the listed
    neurons -- bit-identical spikes, the silent membranes are simply never touched (legal because nothing
    else reads this layer's membrane: it is not the readout layer).

    Requires `plain_delayed_subtract`, thresholds > 0, betas in [0, 1), at most 1000 steps, the current
    produced by a preheader op, and the membrane read by nothing but the LIF itself."""
    if module.num_timesteps > 1000:
        return {}
    cons = consumers(module)
    pre_outputs = {out for op in module.preheader_ops for out in op_outputs(op)}
    plans = {}
    for op in module.loop[0].body:
        if not isinstance(op, LIFUpdateOp):
            continue
        sem = op.semantics
        if (not plain_delayed_subtract(sem) or min(sem.threshold) <= 0.0 or any(b < 0.0 or b >= 1.0 for b in sem.beta)
                or sem.num_neurons > 65535 or op.output_spike == module.output_buffer
                or op.current not in pre_outputs or cons.get(op.membrane) != [op.id]):
            continue
        fp16 = layer_index_of(op.output_spike) if (op.output_spike.startswith("stage")
                                                   and layer_index_of(op.output_spike) in direct_fused) else None
        plans[op.id] = SilentLifPlan(op.id, layer_index_of(op.current), sem.num_neurons, op.current, fp16)
    return plans


def silent_lif_declarations(plans: dict) -> List[str]:
    out = []
    for plan in plans.values():
        out += [f"static uint16_t g_act{plan.idx}[{plan.n}];   /* neurons that can fire (silent-neuron elimination) */",
                f"static int g_nact{plan.idx};"]
    return (["#include <stdint.h>"] + out) if out else []


def silent_lif_preheader(op: LIFUpdateOp, plan: SilentLifPlan) -> List[str]:
    sem = op.semantics
    scalar_beta, scalar_th = len(sem.beta) == 1, len(sem.threshold) == 1
    zero = (f"g_in_f16[{plan.fp16_target}][j] = (float16_t)0.0f" if plan.fp16_target is not None
            else f"{buffer_ref(op.output_spike)}[j] = 0.0f")
    return [
        f"    {{ /* {op.id}: silent-neuron elimination: current is constant over the run, so a neuron below",
        "     * 0.999*threshold*(1-beta) can never fire; list the others once, zero the rest once */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{plan.idx}];",
        f"        const float *cur = {buffer_ref(plan.current)};",
        "        int na = 0;",
        f"        for (int j = 0; j < {plan.n}; j++) {{",
        f"            const float beta = {'L->beta[0]' if scalar_beta else 'L->beta[j]'};",
        f"            const float th = {'L->threshold[0]' if scalar_th else 'L->threshold[j]'};",
        f"            if (!(cur[j] <= 0.999f * th * (1.0f - beta))) {{ g_act{plan.idx}[na++] = (uint16_t)j; }}",
        f"            else {{ {zero}; }}",
        "        }",
        f"        g_nact{plan.idx} = na;",
        "    }",
    ]


def emit_lif_speed(op: LIFUpdateOp) -> List[str]:
    """`emit_lif` with the specialised fast path for layers that qualify (`speed` builds)."""
    if plain_delayed_subtract(op.semantics):
        return emit_lif_fast(op)
    return emit_lif(op)


def emit_lif(op: LIFUpdateOp) -> List[str]:
    """One `LIFUpdateOp` -> a block calling the shared semantic helpers.

    Never hoistable (see `compiler/ir/execution.py::op_is_pure` --
    `LIFUpdateOp` is stateful), so this is only ever emitted inside the
    temporal loop, never in a preheader. Implements NO neuron semantics
    itself -- entirely delegates to `iscas_lif_update()`/
    `iscas_lif_update_inhibition_layer()`/`iscas_param_at()`, whichever
    backend's copy of those (identical algorithm, see each backend's own
    module docstring for why they're not literally shared code).

    B4 Phase B2: the layer index (for `ISCAS_LAYERS[idx]`/current/membrane
    storage) is derived from `op.current`/`op.membrane` -- NEVER from
    `op.output_spike` -- because `SpikeStagingFusionPass` renames a fused
    op's `output_spike` to the downstream `stage{i+1}` buffer it now
    writes directly, which is not this layer's own index. `buffer_ref`
    resolves that write target correctly either way (fused -> a `g_stage`
    slot; unfused -> this layer's own `g_spike` slot) purely from the
    buffer's own name -- no fused/unfused branch needed here at all,
    which is exactly the point: this function computes the SAME final
    emitted spike value regardless of where it's told to put it.
    """
    if op.spike_state is not None:
        return emit_lif_spike_state(op)
    idx = layer_index_of(op.current)
    current_ref = buffer_ref(op.current)
    mem_ref = buffer_ref(op.membrane)
    spike_ref = buffer_ref(op.output_spike)
    return [
        f"    {{ /* {op.id}: lif layer {idx} -> {op.output_spike} */",
        f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        "        if (L->inhibition) {",
        f"            iscas_lif_update_inhibition_layer(L, {current_ref}, {mem_ref}, "
        f"g_inhib_mem_state[{idx}], g_inhib_reset_old[{idx}], {spike_ref});",
        "        } else {",
        "            for (int j = 0; j < L->out_features; j++) {",
        "                float spike, new_mem;",
        f"                iscas_lif_update({current_ref}[j], {mem_ref}[j],",
        "                    iscas_param_at(L->beta, L->beta_len, j),",
        "                    iscas_param_at(L->threshold, L->threshold_len, j),",
        "                    L->reset_mechanism, L->reset_delay,",
        "                    iscas_param_at(L->graded_spikes_factor, L->graded_spikes_factor_len, j),",
        "                    L->state_quant_levels, L->state_quant_num_levels,",
        "                    &spike, &new_mem);",
        f"                {mem_ref}[j] = new_mem;",
        f"                {spike_ref}[j] = spike;",
        "            }",
        "        }",
        f"        g_lif_exec_count[{idx}]++;",
        "    }",
    ]


def emit_schedule(
    module: ExecutionModule,
    *,
    affine_emitter: AffineEmitter,
    input_param: str,
    lif_emitter: LifEmitter = emit_lif,
    input_temporal_macro: str = "ISCAS_INPUT_TEMPORAL",
    input_dim_macro: str = "ISCAS_INPUT_DIM",
    direct_input_source: bool = False,
    absorbed_op_ids: FrozenSet[str] = frozenset(),
    op_epilogue: Optional[Callable[[object], List[str]]] = None,
    gate_binary_affines: bool = False,
    bitmask_delays: bool = True,
    silent_lifs: Optional[dict] = None,
) -> Tuple[List[str], List[str]]:
    """Returns `(preheader_lines, loop_body_lines)`: C statement blocks,
    one per Execution IR op, in program order -- `module.preheader_ops`
    first, then `module.loop[0].body`. This is the ONE place both the
    preheader/loop split AND op-to-C-block translation happen; callers
    (`portable_c.py`, `efr32.py`) only wrap these two line lists into
    their own function/file structure.

    `lif_emitter` defaults to the shared `emit_lif` (identical text for
    every backend). B4 Phase B2.5: EFR32-MVP passes a target-specific
    closure instead, so a compiler-proven-fused edge's `LIFUpdateOp` can
    write its final emitted spike directly into the MVP FP16 input
    representation -- see `compiler/backend/efr32.py`'s own MVP-specific
    emitter functions for that decision; this parameter is the "minimum
    capability hook" mentioned in the B4 Phase B2.5 spec, not a place
    where FP16/MVP knowledge lives itself (the default here has none).

    `direct_input_source` (B4 Phase B3, default `False` -- every existing
    caller's generated text is unaffected): when `True`, a loop-body
    `ConvertOp` reading the model's external input treats `input_param`
    as the single, already-selected timestep vector (see
    `emit_convert`'s own docstring) instead of computing a `t`-indexed
    offset into a whole-tensor buffer. Used by EFR32's streaming
    `iscas_infer_step()` generation only.

    `absorbed_op_ids` (default empty -- every existing caller's generated
    text is unaffected): ids of `ConvertOp`s whose identity copy a backend
    realization of their sole consumer performs itself, reading the
    convert's source directly (EFR32-MVP range-sparse affine, see
    `compiler/backend/efr32.py`). Such an op emits only a comment, never a
    copy loop, and its `g_convert_exec_count` stays 0. No IR is rewritten:
    this is a code-emission decision, like the direct-FP16 LIF hook above.

    `op_epilogue` (default None -- every existing caller's generated text is
    unaffected): lines appended after each op's block, e.g. EFR32's opt-in
    per-stage DWT profiling markers. Emission order and op blocks are unchanged.
    """
    bit_layers = delay_bit_layers(module) if bitmask_delays else {}
    # A staging copy whose only consumer is a bitmask delay push is not needed: the push reads the
    # producer's spike buffer directly.
    push_source = {}
    for push in module.walk_ops():
        if isinstance(push, DelayPushOp) and layer_index_of(push.history) in bit_layers:
            conv = next((o for o in module.walk_ops() if isinstance(o, ConvertOp) and o.output == push.input), None)
            if conv is not None and conv.input != module.input_buffer and consumers(module).get(push.input) == [push.id]:
                push_source[conv.id] = (push.id, buffer_ref(conv.input))
    absorbed_pushes = {pid: src for pid, src in push_source.values()}
    silent_lifs = silent_lifs or {}
    silent_by_current = {plan.current: plan for plan in silent_lifs.values()}
    lif_ops = {o.id: o for o in module.walk_ops() if isinstance(o, LIFUpdateOp)}

    def _emit(ops, in_preheader):
        lines: List[str] = []
        for op in ops:
            if isinstance(op, ConvertOp) and op.id in push_source:
                lines.append(
                    f"    /* {op.id}: convert -> {op.output} ABSORBED: the bitmask delay push reads {op.input} directly */"
                )
            elif isinstance(op, ConvertOp) and op.id in absorbed_op_ids:
                lines.append(
                    f"    /* {op.id}: convert -> {op.output} ABSORBED: its consumer's backend "
                    "realization reads the source directly (identity copy not emitted) */"
                )
            elif isinstance(op, ConvertOp):
                lines.extend(
                    emit_convert(
                        op, module, in_preheader=in_preheader, input_param=input_param,
                        input_temporal_macro=input_temporal_macro, input_dim_macro=input_dim_macro,
                        direct_input_source=direct_input_source,
                    )
                )
            elif isinstance(op, DelayPushOp):
                lines.extend(emit_delay_push_bits(op, absorbed_pushes.get(op.id)) if layer_index_of(op.history) in bit_layers
                             else emit_delay_push(op))
            elif isinstance(op, AffineOp):
                idx = layer_index_of(op.output)
                if op.weight_layer.synapse_delays is not None and idx in bit_layers:
                    lines.extend(emit_affine_delay_bits(op, idx, bit_layers[idx]))
                elif (gate_binary_affines and op.weight_layer.synapse_delays is None
                      and op.input != module.input_buffer and binary_spike_buffer(module, op.input)):
                    lines.extend(emit_affine_spike_gated(op, idx))
                else:
                    lines.extend(affine_emitter(op, idx))
            elif isinstance(op, LIFUpdateOp) and op.id in silent_lifs:
                plan = silent_lifs[op.id]
                lines.extend(emit_lif_fast(op, plan.fp16_target, active=(f"g_act{plan.idx}", f"g_nact{plan.idx}")))
            elif isinstance(op, LIFUpdateOp):
                lines.extend(lif_emitter(op))
            if in_preheader:
                for out in op_outputs(op):
                    if out in silent_by_current:
                        plan = silent_by_current[out]
                        lines.extend(silent_lif_preheader(lif_ops[plan.lif_id], plan))
            if op_epilogue is not None:
                lines.extend(op_epilogue(op))
        return lines

    preheader_lines = _emit(module.preheader_ops, True)
    loop_lines = _emit(module.loop[0].body, False)
    return preheader_lines, loop_lines
