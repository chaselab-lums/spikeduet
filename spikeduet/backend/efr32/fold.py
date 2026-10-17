"""Temporal output-affine folding (``fold_output_affine=True``).

The readout sums the final layer's membrane over all T steps. When the output LIF has beta 0 for
every neuron, no reset, no inhibition and no state quantization, and nothing reads its spikes,
``U_out[t] = W x[t] + b`` and ``sum_t U_out[t] = W (sum_t x[t]) + T b``. The output affine then
runs once, in ``iscas_infer_readout()``, on the time-summed input; each step only accumulates.
This is a backend schedule transform: the IR is unchanged.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from spikeduet.backend.efr32.cpu_emitters import SparseCpuPlan
from spikeduet.backend.efr32.profiling import StageProfile
from spikeduet.backend.schedule_common import buffer_ref, emit_lif, layer_index_of
from spikeduet.ir.execution import AffineOp, ExecutionModule, LIFUpdateOp
from spikeduet.ir.execution_analysis import consumers, producers



@dataclass(frozen=True)
class OutputFoldPlan:
    """A legal fold of the output affine."""

    layer: int                  # output affine / LIF layer index
    affine_id: str
    lif_id: str
    acc_buffer: str             # Execution-IR buffer that holds the time-summed affine input
    producer_lif_id: Optional[str]   # LIFUpdateOp that accumulates into acc_buffer (fused), else None (copy loop)
    rows: int
    cols: int
    has_bias: bool
    steps: int


def output_fold_plan(module: ExecutionModule) -> OutputFoldPlan:
    """Legality check for temporal output-affine folding; raises ValueError naming the first violated condition."""
    body = module.loop[0].body
    ops = {op.id: op for op in module.walk_ops()}
    prod, cons = producers(module), consumers(module)
    lifs = [op for op in body if isinstance(op, LIFUpdateOp)]
    if not lifs:
        raise ValueError("fold_output_affine: no LIF layer in the temporal loop")
    out_lif = lifs[-1]
    sem = out_lif.semantics
    if layer_index_of(out_lif.current) != max(layer_index_of(op.current) for op in lifs):
        raise ValueError("fold_output_affine: the last loop LIF is not the final layer")
    if sem.reset_mechanism != "none":
        raise ValueError(f"fold_output_affine: output reset_mechanism is {sem.reset_mechanism!r}, needs 'none'")
    if any(b != 0.0 for b in sem.beta):
        raise ValueError("fold_output_affine: output beta must be exactly 0 for every neuron (fixed); learned/nonzero beta is not supported")
    if sem.inhibition or sem.state_quant is not None:
        raise ValueError("fold_output_affine: output layer uses inhibition or state quantization")
    if cons.get(out_lif.output_spike):
        raise ValueError("fold_output_affine: the output spikes are consumed by another op")
    if any(c != out_lif.id for c in cons.get(out_lif.membrane, [])):
        raise ValueError("fold_output_affine: the output membrane is read by another op")
    aff = ops.get(prod.get(out_lif.current))
    if not isinstance(aff, AffineOp) or aff.id not in {op.id for op in body}:
        raise ValueError("fold_output_affine: the output current is not produced by a loop-resident AffineOp")
    if cons.get(aff.output, []) != [out_lif.id]:
        raise ValueError("fold_output_affine: the output affine's result has another consumer")
    src = ops.get(prod.get(aff.input))
    fused = (isinstance(src, LIFUpdateOp) and not src.semantics.inhibition and src.id in {op.id for op in body}
             and cons.get(aff.input, []) == [aff.id])
    w = aff.weight_layer
    return OutputFoldPlan(layer=layer_index_of(aff.output), affine_id=aff.id, lif_id=out_lif.id, acc_buffer=aff.input,
                          producer_lif_id=src.id if fused else None, rows=w.out_features, cols=w.in_features,
                          has_bias=w.bias is not None, steps=module.num_timesteps)


def fold_lif_emitter(plan: OutputFoldPlan, inner=None):
    """The producer LIF accumulates its emitted spikes into the folded affine's input buffer (`+=`); every other LIF
    is emitted unchanged; the output LIF is not executed per step (its value is not needed)."""
    inner = inner or emit_lif

    def emitter(op: LIFUpdateOp) -> List[str]:
        if op.id == plan.lif_id:
            return [f"    /* {op.id}: output LIF FOLDED (beta 0, no reset): U_out[t] = W x[t] + b is formed once, "
                    "in iscas_infer_readout() */"]
        lines = inner(op)
        if op.id == plan.producer_lif_id:
            spike_ref = buffer_ref(op.output_spike)
            old = f"                {spike_ref}[j] = spike;"
            assert lines.count(old) == 1
            lines = [l if l != old else f"                {spike_ref}[j] += spike;   /* folded output affine: sum over time */"
                     for l in lines]
        return lines

    return emitter


def fold_affine_emitter(plan: OutputFoldPlan, inner):
    def emitter(op: AffineOp, idx: int) -> List[str]:
        if op.id == plan.affine_id:
            lines = [f"    /* {op.id}: output affine FOLDED: executed once on the time-summed input in iscas_infer_readout() */"]
            if plan.producer_lif_id is None:
                ref = buffer_ref(op.input)
                lines += [f"    for (int i = 0; i < {plan.cols}; i++) {{ g_fold_acc[i] += {ref}[i]; }}   /* sum over time */"]
            return lines
        return inner(op, idx)

    return emitter


def fold_readout_lines(plan: OutputFoldPlan, cpu_plans: Dict[int, SparseCpuPlan], prof: StageProfile) -> List[str]:
    """The one output-affine execution in the readout: ``g_mem_sum = W acc + T b``."""
    l, m, n = plan.layer, plan.rows, plan.cols
    acc = "g_fold_acc" if plan.producer_lif_id is None else buffer_ref(plan.acc_buffer)
    out = [f"    {{   /* folded output affine (layer {l}), once per inference */"]
    if l in cpu_plans:
        out += ["        int all_ones;",
                f"        const int k = iscas_csp_scan({acc}, {n}, g_csp_idx[{l}], &all_ones);",
                f"        iscas_csp_acc{l}(g_csp_wT[{l}], g_csp_idx[{l}], k, all_ones ? 0 : {acc}, g_current[{l}]);   /* general (count) path */",
                f"        g_csp_event_count[{l}] += k;"]
    else:
        out += [f"        const iscas_layer_t *L = &ISCAS_LAYERS[{l}];",
                f"        for (int j = 0; j < {m}; j++) {{",
                "            float c = L->bias ? L->bias[j] : 0.0f;",
                f"            for (int i = 0; i < {n}; i++) {{ c += L->weight[(size_t)j * {n} + i] * {acc}[i]; }}",
                f"            g_current[{l}][j] = c;",
                "        }"]
    out += ["#ifdef ISCAS_SPARSE_VERIFY",
            "        {   /* correctness build only: dense FP32 recompute of W acc + b, bitwise compare */",
            f"            const iscas_layer_t *LV = &ISCAS_LAYERS[{l}];",
            "            int bad = 0;",
            f"            for (int j = 0; j < {m}; j++) {{",
            "                float c = LV->bias ? LV->bias[j] : 0.0f;",
            f"                for (int i = 0; i < {n}; i++) {{ c += LV->weight[(size_t)j * {n} + i] * {acc}[i]; }}",
            f"                bad |= memcmp(&c, &g_current[{l}][j], sizeof(float)) != 0;",
            "            }",
            f"            g_csp_verify_mismatch[{l}] += bad;",
            "        }",
            "#endif"] if l in cpu_plans else []
    bias = f"ISCAS_LAYERS[{l}].bias"
    out += [f"        for (int j = 0; j < {m}; j++) {{   /* sum_t (W x[t] + b) = (W acc + b) + (T-1) b */",
            f"            g_mem_sum[j] = g_current[{l}][j]" + (f" + {float(plan.steps - 1)!r}f * {bias}[j];" if plan.has_bias else ";"),
            "        }",
            f"        g_affine_exec_count[{l}]++;",
            "    }",
            *prof.lap("fold.affine")]
    return out
