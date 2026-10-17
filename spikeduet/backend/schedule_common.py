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
from typing import FrozenSet, Callable, List, Optional, Tuple

from spikeduet.ir.execution import AffineOp, ConvertOp, ExecutionModule, LIFUpdateOp

AffineEmitter = Callable[[AffineOp, int], List[str]]
LifEmitter = Callable[[LIFUpdateOp], List[str]]


def layer_index_of(buffer_name: str) -> int:
    """Trailing layer index of a lowered buffer name (``"current2"`` -> ``2``). Passes move ops
    but never rename buffers, so this index is also the layer's position in ``ISCAS_LAYERS``."""
    m = re.search(r"(\d+)$", buffer_name)
    if not m:
        raise ValueError(f"buffer name {buffer_name!r} has no trailing layer index")
    return int(m.group(1))


# Fusion makes a LIF write ``stage{i+1}``, so a buffer's storage is derived from its own name only.
_ARRAY_PREFIXES = {
    "stage": "g_stage",
    "current": "g_current",
    "mem": "g_mem",
    "spike": "g_spike",
}


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
    """Scalar C affine: start from the bias, accumulate row by row."""
    stage_ref = buffer_ref(op.input)
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
    """EFR32 MVP affine: cast the staged input to FP16, one
    ``sl_math_mvp_matrix_vector_mult_f16()`` call, then add the FP32 bias. Used for edges whose
    upstream LIF does not write FP16 directly."""
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


def emit_lif(op: LIFUpdateOp) -> List[str]:
    """A block calling the shared LIF helpers. The layer index comes from ``op.current``,
    because a fused LIF's ``output_spike`` names the next layer's staging buffer."""
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
) -> Tuple[List[str], List[str]]:
    """Returns ``(preheader_lines, loop_body_lines)``, one C block per op in program order.

    Args:
        affine_emitter: realization of each ``AffineOp``.
        lif_emitter: realization of each ``LIFUpdateOp`` (default ``emit_lif``).
        input_param: C expression of the external input pointer.
        direct_input_source: see ``emit_convert``.
        absorbed_op_ids: convert ops whose consumer reads the source itself; they emit only a
            comment.
        op_epilogue: extra lines after each op's block (e.g. profiling markers).
    """
    def _emit(ops, in_preheader):
        lines: List[str] = []
        for op in ops:
            if isinstance(op, ConvertOp) and op.id in absorbed_op_ids:
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
            elif isinstance(op, AffineOp):
                idx = layer_index_of(op.output)
                lines.extend(affine_emitter(op, idx))
            elif isinstance(op, LIFUpdateOp):
                lines.extend(lif_emitter(op))
            if op_epilogue is not None:
                lines.extend(op_epilogue(op))
        return lines

    preheader_lines = _emit(module.preheader_ops, True)
    loop_lines = _emit(module.loop[0].body, False)
    return preheader_lines, loop_lines
