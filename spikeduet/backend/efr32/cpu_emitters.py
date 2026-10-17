"""Event-driven Cortex-M33 realization of ``AffineOp`` (FP32).

For every active input column, the column of W is accumulated into the output instead of
executing the zero MACs. Up to 16 accumulators stay in FPU registers per pass, and W is stored
transposed so a column is contiguous. With binary inputs the multiply is skipped
(``acc + w == fma(w, 1, acc)`` exactly). Additions happen in ascending column order starting from
the bias, so results are bitwise equal to the dense scalar loop (checked per step by
``ISCAS_SPARSE_VERIFY`` builds). All memory is allocated in ``snn_init()``.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from spikeduet.backend.c_common import float_literal
from spikeduet.backend.efr32.cpu_sparse import (
    CPU_SPARSE_HELPERS,
    CPU_SPARSE_WEIGHT_PLACEMENTS,
    cpu_sparse_init_lines,
    cpu_sparse_layer_definitions,
    cpu_sparse_pinned_definitions,
    cpu_sparse_pinned_init_lines,
)
from spikeduet.backend.efr32.profiling import NO_PROFILE, StageProfile
from spikeduet.backend.schedule_common import buffer_ref, layer_index_of
from spikeduet.ir.execution import AffineOp, ConvertOp, ExecutionModule
from spikeduet.ir.execution_analysis import consumers, producers


@dataclass(frozen=True)
class SparseCpuPlan:
    """One event-driven Cortex-M33 layer."""
    layer: int
    rows: int
    cols: int
    source: str                      # C expression of the FP32 vector the scan reads
    absorbed_convert: Optional[str]  # ConvertOp whose identity copy is absorbed (not emitted)
    has_bias: bool


def cpu_sparse_plan(module: ExecutionModule, layers: Optional[Set[int]] = None) -> Dict[int, SparseCpuPlan]:
    """Loop-body AffineOps realized event-driven on the Cortex-M33 (`layers=None`: all of them).
    A hoisted preheader affine runs once and is never a candidate. The scan reads the ConvertOp's
    source directly when that ConvertOp's copy has no other consumer (the copy is absorbed), and
    otherwise the affine's own FP32 input buffer (e.g. a fused LIF's `g_stage[i]`)."""
    loop_ids = {op.id for op in module.loop[0].body}
    ops_by_id = {op.id: op for op in module.walk_ops()}
    prod, cons = producers(module), consumers(module)
    plans: Dict[int, SparseCpuPlan] = {}
    for op in module.loop[0].body:
        if not isinstance(op, AffineOp):
            continue
        idx = layer_index_of(op.output)
        if layers is not None and idx not in layers:
            continue
        conv = ops_by_id.get(prod.get(op.input))
        absorbed, source = None, buffer_ref(op.input)
        if isinstance(conv, ConvertOp) and conv.id in loop_ids and cons.get(op.input, []) == [op.id]:
            absorbed = conv.id
            source = "step_input" if conv.input == module.input_buffer else buffer_ref(conv.input)
        w = op.weight_layer
        plans[idx] = SparseCpuPlan(layer=idx, rows=w.out_features, cols=w.in_features, source=source,
                                   absorbed_convert=absorbed, has_bias=w.bias is not None)
    return plans


def cpu_layer_definitions(plan: SparseCpuPlan, weight, with_kernel: bool = True) -> List[str]:
    """Transposed weight table and accumulate kernel for one layer. The literals use the same
    format as model_data.h, so the weights are the same floats."""
    return cpu_sparse_layer_definitions(plan.layer, plan.rows, plan.cols, weight.detach().cpu().float().tolist(),
                                        f"iscas_layer{plan.layer}_bias" if plan.has_bias else None, float_literal,
                                        with_kernel=with_kernel)


def cpu_affine_emitter(plans: Dict[int, SparseCpuPlan], fallback, prof: StageProfile = NO_PROFILE,
                               pinned: FrozenSet[int] = frozenset()):
    """Event-driven emitter; other layers go to ``fallback``. Layers in ``pinned`` use the
    pointer-table kernel that reads pinned columns from SRAM."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        plan = plans.get(idx)
        if plan is None:
            return fallback(op, idx)
        if prof.enabled:
            prof.substaged.add(idx)
        n = plan.cols
        absorbed = f"; {plan.absorbed_convert} absorbed" if plan.absorbed_convert else ""
        return [
            f"    {{ /* {op.id}: affine layer {idx} (Cortex-M33 event-driven sparse, FP32{absorbed}) */",
            f"        const float *src = {plan.source};",
            "        int all_ones;",
            f"        const int k = iscas_csp_scan(src, {n}, g_csp_idx[{idx}], &all_ones);",
            *prof.lap(f"affine{idx}.plan", "        "),
            (f"        iscas_csp_accp{idx}(g_csp_colp{idx}, g_csp_idx[{idx}], k, all_ones ? 0 : src, g_current[{idx}]);"
             if idx in pinned else
             f"        iscas_csp_acc{idx}(g_csp_wT[{idx}], g_csp_idx[{idx}], k, all_ones ? 0 : src, g_current[{idx}]);"),
            f"        g_csp_event_count[{idx}] += k;",
            f"        if (k == 0) {{ g_csp_empty_count[{idx}]++; }}",
            f"        if (!all_ones) {{ g_csp_nonbinary_count[{idx}]++; }}",
            *prof.lap(f"affine{idx}.compute", "        "),
            "#ifdef ISCAS_SPARSE_VERIFY",
            "        {   /* correctness build only (never timed): dense FP32 recompute exactly as the scalar realization, bitwise compare */",
            f"            const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
            "            int bad = 0;",
            "            for (int j = 0; j < L->out_features; j++) {",
            "                float c = L->bias ? L->bias[j] : 0.0f;",
            "                for (int i = 0; i < L->in_features; i++) {",
            "                    c += L->weight[(size_t)j * L->in_features + i] * src[i];",
            "                }",
            f"                bad |= memcmp(&c, &g_current[{idx}][j], sizeof(float)) != 0;",
            "            }",
            f"            g_csp_verify_mismatch[{idx}] += bad;",
            "        }",
            "#endif",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
        ]

    return emitter


def cpu_declarations(module: ExecutionModule, plans: Dict[int, SparseCpuPlan],
                             pinned_columns: Optional[Dict[int, Sequence[int]]] = None) -> List[str]:
    if not plans:
        return []
    ops = {layer_index_of(op.output): op for op in module.walk_ops() if isinstance(op, AffineOp)}
    out = [CPU_SPARSE_HELPERS]
    for idx, p in sorted(plans.items()):
        pin = bool(pinned_columns) and idx in pinned_columns
        out += cpu_layer_definitions(p, ops[idx].weight_layer.weight, with_kernel=not pin)
        if pin:
            out += cpu_sparse_pinned_definitions(idx, p.rows, p.cols, pinned_columns[idx],
                                                 f"iscas_layer{idx}_bias" if p.has_bias else None)
    return out


def check_weight_placement(plans: Dict[int, SparseCpuPlan], placement: Optional[Dict[int, str]]) -> None:
    """``cpu_sparse_weight_placement`` must name exactly the event-driven layers."""
    if placement is None:
        return
    if set(placement) != set(plans):
        raise ValueError(f"cpu_sparse_weight_placement must list exactly the event-driven M33 layers {sorted(plans)}, "
                         f"got {sorted(placement)}")
    bad = {l: v for l, v in placement.items() if v not in CPU_SPARSE_WEIGHT_PLACEMENTS}
    if bad:
        raise ValueError(f"cpu_sparse_weight_placement values must be in {CPU_SPARSE_WEIGHT_PLACEMENTS}: {bad}")


def cpu_init_lines(plans: Dict[int, SparseCpuPlan], placement: Optional[Dict[int, str]] = None,
                           pinned_columns: Optional[Dict[int, Sequence[int]]] = None) -> List[str]:
    out: List[str] = []
    for idx, p in sorted(plans.items()):
        out += cpu_sparse_init_lines(idx, p.cols, None if placement is None else placement[idx])
        if pinned_columns and idx in pinned_columns:
            out += cpu_sparse_pinned_init_lines(idx, p.rows, p.cols, len(set(pinned_columns[idx])))
    return out


def check_pinned_columns(plans: Dict[int, SparseCpuPlan], placement: Optional[Dict[int, str]],
                          pinned_columns: Optional[Dict[int, Sequence[int]]]) -> FrozenSet[int]:
    """Pinning applies only to event-driven layers whose table is placed in flash."""
    if not pinned_columns:
        return frozenset()
    bad = set(pinned_columns) - set(plans)
    if bad:
        raise ValueError(f"cpu_sparse_pinned_columns names layers {sorted(bad)} that are not event-driven M33 layers")
    for l in pinned_columns:
        if placement is None or placement.get(l) != "flash":
            raise ValueError(f"pinned layer {l} needs cpu_sparse_weight_placement[{l}] == 'flash' (pinning keeps the table in flash)")
        cols = set(int(c) for c in pinned_columns[l])
        if not cols or min(cols) < 0 or max(cols) >= plans[l].cols:
            raise ValueError(f"pinned columns of layer {l} must be a non-empty subset of 0..{plans[l].cols - 1}")
    return frozenset(pinned_columns)


CPU_COUNTERS = ("event", "empty", "nonbinary")


def cpu_reset_lines(plans: Dict[int, SparseCpuPlan]) -> List[str]:
    if not plans:
        return []
    out = [f"    memset(g_csp_{c}_count, 0, sizeof(g_csp_{c}_count));" for c in CPU_COUNTERS]
    out.append("    memset(g_csp_verify_mismatch, 0, sizeof(g_csp_verify_mismatch));")
    return out


def cpu_accessor_lines(plans: Dict[int, SparseCpuPlan]) -> List[str]:
    if not plans:
        return []
    out = [f"int iscas_debug_cpu_sparse_{c}_count(int layer) {{ return g_csp_{c}_count[layer]; }}" for c in CPU_COUNTERS]
    out += [
        "int iscas_debug_cpu_sparse_verify_mismatch(int layer) { return g_csp_verify_mismatch[layer]; }",
        "int iscas_debug_cpu_sparse_layer(int layer) { switch (layer) { "
        + " ".join(f"case {i}: return 1;" for i in sorted(plans)) + " default: return 0; } }",
    ]
    return out
