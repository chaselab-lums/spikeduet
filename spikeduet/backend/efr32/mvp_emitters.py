"""MVP realizations of ``AffineOp``: dense with direct-FP16 spike input, range-sparse, one raw
full-width launch, and the lean or windowed dense input layer."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from spikeduet.backend.efr32.cost_model import (
    EFR32_SPARSE_MIN_LAYER_GAIN,
    mvp_dense_affine_cycles,
    mvp_sparse_affine_cycles,
    sparse_merge_gap_pairs,
)
from spikeduet.backend.efr32.mvp_sparse import (
    EFR32_MVP_LAUNCH_CYCLES,
    EFR32_MVP_MAX_DIM,
    EFR32_MVP_SPARSE_MIN_LAUNCHES,
    SPARSE_MVP_HELPERS,
)
from spikeduet.backend.efr32.profiling import NO_PROFILE, StageProfile
from spikeduet.backend.schedule_common import buffer_ref, emit_lif, layer_index_of, mvp_affine_emitter
from spikeduet.ir.execution import AffineOp, ConvertOp, ExecutionModule, LIFUpdateOp
from spikeduet.ir.execution_analysis import consumers, producers


def mvp_direct_fused_layer_indices(module: ExecutionModule) -> Set[int]:
    """Affine layers whose input is written directly by a non-inhibited LIF (fusion ran).

    For these the LIF can store FP16 spikes into the MVP input and skip the staging loop. Inhibited
    LIFs are excluded: their shared two-pass helper writes an FP32 output vector, so those edges
    keep the staged path.
    """
    ops_by_id = {op.id: op for op in module.walk_ops()}
    buffer_producers = producers(module)

    result: Set[int] = set()
    for op in module.walk_ops():
        if isinstance(op, AffineOp):
            producer_id = buffer_producers.get(op.input)
            producer_op = ops_by_id.get(producer_id) if producer_id is not None else None
            if isinstance(producer_op, LIFUpdateOp) and not producer_op.semantics.inhibition:
                result.add(layer_index_of(op.output))
    return result


def mvp_direct_affine_emitter(direct_fused: Set[int]):
    """Dense MVP affine; layers in ``direct_fused`` skip the FP32->FP16 staging loop."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        if idx not in direct_fused:
            return mvp_affine_emitter(op, idx)
        return [
            f"    {{ /* {op.id}: affine layer {idx} (MVP, direct-FP16 input -- fused, no staging loop) */",
            f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
            f"        sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_cur_f16[{idx}]);",
            "        for (int j = 0; j < L->out_features; j++) {",
            f"            g_current[{idx}][j] = (float)g_cur_f16[{idx}][j] + (L->bias ? L->bias[j] : 0.0f);",
            "        }",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
        ]

    return emitter


def mvp_direct_lif_emitter(direct_fused: Set[int]):
    """LIF emitter whose output feeds a ``direct_fused`` layer: same ``iscas_lif_update()`` call
    as ``emit_lif``, but the spike is stored as FP16 into the next layer's MVP input."""

    def emitter(op: LIFUpdateOp) -> List[str]:
        target_idx: Optional[int] = None
        if op.output_spike.startswith("stage"):
            candidate = layer_index_of(op.output_spike)
            if candidate in direct_fused:
                target_idx = candidate

        if target_idx is None:
            return emit_lif(op)

        idx = layer_index_of(op.current)
        current_ref = buffer_ref(op.current)
        mem_ref = buffer_ref(op.membrane)
        return [
            f"    {{ /* {op.id}: lif layer {idx} -> direct FP16 g_in_f16[{target_idx}] (MVP-fused) */",
            f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
            "        for (int j = 0; j < L->out_features; j++) {",
            "            float spike, new_mem;",
            f"            iscas_lif_update({current_ref}[j], {mem_ref}[j],",
            "                iscas_param_at(L->beta, L->beta_len, j),",
            "                iscas_param_at(L->threshold, L->threshold_len, j),",
            "                L->reset_mechanism, L->reset_delay,",
            "                iscas_param_at(L->graded_spikes_factor, L->graded_spikes_factor_len, j),",
            "                L->state_quant_levels, L->state_quant_num_levels,",
            "                &spike, &new_mem);",
            f"            {mem_ref}[j] = new_mem;",
            f"            g_in_f16[{target_idx}][j] = (float16_t)spike;",
            "        }",
            f"        g_lif_exec_count[{idx}]++;",
            f"        g_mvp_direct_fp16_write_count[{target_idx}]++;",
            "    }",
        ]

    return emitter


@dataclass(frozen=True)
class SparseMvpPlan:
    """Per-layer range-sparse realization decision (compile time)."""
    layer: int
    rows: int                        # M = out_features
    cols: int                        # N = in_features (even)
    gap_pairs: int                   # merge two active runs if <= this many inactive pairs separate them
    source: str                      # C expression of the FP32 vector the sparse scan reads
    absorbed_convert: Optional[str]  # ConvertOp id whose identity copy is absorbed (not emitted)
    fp16_source: bool = False        # input already FP16 in g_in_f16 (direct-FP16 fused LIF spikes): no staging


def mvp_sparse_plan(
    module: ExecutionModule,
    *,
    activity: Optional[Dict[int, Tuple[float, float]]] = None,
    force_layers: FrozenSet[int] = frozenset(),
) -> Dict[int, SparseMvpPlan]:
    """Affine layers that get the range-sparse MVP realization.

    An FP32-staged layer is eligible when it runs inside the loop (a hoisted affine runs once),
    its input comes from a ``ConvertOp``, ``in_features`` is even, the dimensions fit the MVP's
    10-bit fields, and its dense work exceeds ``EFR32_MVP_SPARSE_MIN_LAUNCHES`` launches (smaller
    layers are launch-bound).

    A hidden edge with FP16 spikes in place is selected only when ``activity``
    (``{layer: (mean launches, mean processed pairs) per step}``) predicts a saving of at least
    ``EFR32_SPARSE_MIN_LAYER_GAIN``, or when it is in ``force_layers``. ``activity`` also vetoes
    FP32 layers whose predicted saving is too small.
    """
    loop_ids = {op.id for op in module.loop[0].body}
    ops_by_id = {op.id: op for op in module.walk_ops()}
    prod = producers(module)
    cons = consumers(module)
    direct_fused = mvp_direct_fused_layer_indices(module)
    plans: Dict[int, SparseMvpPlan] = {}

    def profile_says_sparse(idx: int, rows: int, cols: int, fp32: bool) -> Optional[bool]:
        if activity is None or idx not in activity:
            return None
        launches, pairs = activity[idx]
        dense = mvp_dense_affine_cycles(rows, cols, fp32_source=fp32)
        sparse = mvp_sparse_affine_cycles(rows, cols, launches, pairs, fp32_source=fp32)
        return sparse <= (1.0 - EFR32_SPARSE_MIN_LAYER_GAIN) * dense

    for op in module.loop[0].body:
        if not isinstance(op, AffineOp):
            continue
        idx = layer_index_of(op.output)
        if idx in direct_fused:
            rows, cols = op.weight_layer.out_features, op.weight_layer.in_features
            if cols % 2 or rows > EFR32_MVP_MAX_DIM or cols // 2 > EFR32_MVP_MAX_DIM:
                continue
            if idx in force_layers or profile_says_sparse(idx, rows, cols, fp32=False):
                plans[idx] = SparseMvpPlan(
                    layer=idx, rows=rows, cols=cols, gap_pairs=sparse_merge_gap_pairs(rows, fp16_source=True),
                    source=f"g_in_f16[{idx}]", absorbed_convert=None, fp16_source=True,
                )
            continue
        conv = ops_by_id.get(prod.get(op.input))
        if not isinstance(conv, ConvertOp) or conv.id not in loop_ids:
            continue
        rows, cols = op.weight_layer.out_features, op.weight_layer.in_features
        pairs = cols // 2
        if cols % 2 or rows > EFR32_MVP_MAX_DIM or pairs > EFR32_MVP_MAX_DIM:
            continue
        if idx not in force_layers:
            if rows * pairs < EFR32_MVP_SPARSE_MIN_LAUNCHES * EFR32_MVP_LAUNCH_CYCLES:
                continue
            if profile_says_sparse(idx, rows, cols, fp32=True) is False:
                continue
        sole = cons.get(op.input, []) == [op.id]
        if sole:
            source = "step_input" if conv.input == module.input_buffer else buffer_ref(conv.input)
        else:
            source = buffer_ref(op.input)
        plans[idx] = SparseMvpPlan(
            layer=idx, rows=rows, cols=cols, gap_pairs=sparse_merge_gap_pairs(rows),
            source=source, absorbed_convert=conv.id if sole else None,
        )
    return plans


def mvp_range_affine_emitter(plans: Dict[int, SparseMvpPlan], fallback, prof: StageProfile = NO_PROFILE):
    """Range-sparse emitter: per step, merge active column pairs into ranges and launch the MVP on
    each. The input is FP32 (zero test on the float bits, cast per processed pair) or FP16 spikes in
    place. Other layers go to ``fallback``."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        plan = plans.get(idx)
        if plan is None:
            return fallback(op, idx)
        if prof.enabled:
            prof.substaged.add(idx)
        m, npr = plan.rows, plan.cols // 2
        fp16 = plan.fp16_source
        absorbed = f"; {plan.absorbed_convert} absorbed" if plan.absorbed_convert else ""
        src_note = "; FP16 source in place (fused LIF output), no staging" if fp16 else ""
        stage_cost = "0" if fp16 else "ISCAS_SP_STAGE_CYCLES_PER_PAIR"
        lines = [
            f"    {{ /* {op.id}: affine layer {idx} (MVP, range-merged sparse, gap {plan.gap_pairs} pairs{absorbed}{src_note}) */",
            f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        ]
        if fp16:
            lines += [
                f"        const uint32_t *xb = (const uint32_t *)g_in_f16[{idx}];  /* one FP16 pair per word; +0 pairs are all-zero bits */",
            ]
        else:
            lines += [
                f"        const float *src = {plan.source};",
                "        const uint32_t *xb = (const uint32_t *)src;  /* +0.0f is all-zero bits; -0.0f counts as active (extra work only) */",
            ]
        test = "xb[p] != 0u" if fp16 else "(xb[2 * p] | xb[2 * p + 1]) != 0u"
        lines += [
            f"        uint16_t *rs = g_sparse_rs[{idx}], *re = g_sparse_re[{idx}];",
            "        int nr = 0, span = 0;",
            f"        for (int p = 0; p < {npr}; p++) {{",
            f"            if ({test}) {{",
            f"                if (nr > 0 && p - (int)re[nr - 1] <= {plan.gap_pairs}) {{ span += p + 1 - (int)re[nr - 1]; re[nr - 1] = (uint16_t)(p + 1); }}",
            "                else { rs[nr] = (uint16_t)p; re[nr] = (uint16_t)(p + 1); nr++; span++; }",
            "            }",
            "        }",
            f"        g_sparse_range_count[{idx}] += nr;",
            *prof.lap(f"affine{idx}.plan", "        "),
            "        if (nr == 0) {",
            f"            g_sparse_empty_count[{idx}]++;",
            f"            for (int j = 0; j < {m}; j++) {{ g_current[{idx}][j] = 0.0f + (L->bias ? L->bias[j] : 0.0f); }}",
            "        } else {",
            "            /* runtime guard (measured cost model): never pay more launches than one dense launch */",
            f"            if ((int32_t)nr * ISCAS_SP_LAUNCH_CYCLES + (int32_t)span * ({m} + {stage_cost})",
            f"                >= ISCAS_SP_LAUNCH_CYCLES + (int32_t){npr} * ({m} + {stage_cost})) {{",
            f"                nr = 1; rs[0] = 0; re[0] = {npr}; span = {npr}; g_sparse_fallback_count[{idx}]++;",
            "            }",
        ]
        if not fp16:
            lines += [
                "            for (int r = 0; r < nr; r++) {",
                f"                for (int i = 2 * (int)rs[r]; i < 2 * (int)re[r]; i++) {{ g_in_f16[{idx}][i] = (float16_t)src[i]; }}",
                "            }",
            ]
        lines += [
            *prof.lap(f"affine{idx}.stage", "            "),
            "            iscas_sp_enter();",
            f"            iscas_sp_setup({m}, {npr}, g_cur_f16[{idx}], g_sparse_acc[{idx}]);",
            "            for (int r = 0; r < nr; r++) {",
            f"                iscas_sp_launch(g_weight_f16[{idx}], g_in_f16[{idx}], rs[r], re[r] - rs[r], r == 0, r == nr - 1);",
            "            }",
            "            iscas_sp_exit();",
            f"            g_sparse_launch_count[{idx}] += nr;",
            f"            g_sparse_pair_count[{idx}] += span;",
            *prof.lap(f"affine{idx}.mvp", "            "),
            f"            for (int j = 0; j < {m}; j++) {{",
            f"                g_current[{idx}][j] = (float)g_cur_f16[{idx}][j] + (L->bias ? L->bias[j] : 0.0f);",
            "            }",
            "        }",
            *prof.lap(f"affine{idx}.out", "        "),
            "#ifdef ISCAS_SPARSE_VERIFY",
            "        {   /* correctness build only (never timed): recompute with the deployed dense SDK kernel, compare bitwise */",
        ]
        if not fp16:
            lines.append(f"            for (int i = 0; i < {plan.cols}; i++) {{ g_in_f16[{idx}][i] = (float16_t)src[i]; }}")
        lines += [
            f"            sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_sparse_ref[{idx}]);",
            f"            const uint16_t *a = (const uint16_t *)g_sparse_ref[{idx}];",
            f"            const uint16_t *b = (const uint16_t *)g_cur_f16[{idx}];",
            "            int bad = 0;",
            f"            for (int j = 0; j < {m}; j++) {{ bad |= (nr == 0) ? (a[j] != 0u) : (a[j] != b[j]); }}",
            f"            g_sparse_verify_mismatch[{idx}] += bad;",
            "        }",
            "#endif",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
        ]
        return lines

    return emitter


def mvp_full_width_plans(module: ExecutionModule, layers: FrozenSet[int]) -> Dict[int, SparseMvpPlan]:
    """Dense ``layers`` through one full-width launch of the raw range kernel instead of the SDK
    call (no scan, no sparsity). Separates SDK-call overhead from sparsity gains."""
    loop_ids = {op.id for op in module.loop[0].body}
    ops_by_id = {op.id: op for op in module.walk_ops()}
    prod, cons = producers(module), consumers(module)
    direct_fused = mvp_direct_fused_layer_indices(module)
    plans: Dict[int, SparseMvpPlan] = {}
    for op in module.loop[0].body:
        if not isinstance(op, AffineOp) or layer_index_of(op.output) not in layers:
            continue
        idx = layer_index_of(op.output)
        rows, cols = op.weight_layer.out_features, op.weight_layer.in_features
        if cols % 2 or rows > EFR32_MVP_MAX_DIM or cols // 2 > EFR32_MVP_MAX_DIM:
            raise ValueError(f"layer {idx} does not fit the raw MVP kernel")
        if idx in direct_fused:
            plans[idx] = SparseMvpPlan(layer=idx, rows=rows, cols=cols, gap_pairs=0, source=f"g_in_f16[{idx}]",
                                       absorbed_convert=None, fp16_source=True)
            continue
        conv = ops_by_id.get(prod.get(op.input))
        sole = isinstance(conv, ConvertOp) and conv.id in loop_ids and cons.get(op.input, []) == [op.id]
        source = ("step_input" if conv.input == module.input_buffer else buffer_ref(conv.input)) if sole else buffer_ref(op.input)
        plans[idx] = SparseMvpPlan(layer=idx, rows=rows, cols=cols, gap_pairs=0, source=source,
                                   absorbed_convert=conv.id if sole else None)
    return plans


def mvp_full_width_affine_emitter(plans: Dict[int, SparseMvpPlan], fallback, prof: StageProfile = NO_PROFILE):
    def emitter(op: AffineOp, idx: int) -> List[str]:
        plan = plans.get(idx)
        if plan is None:
            return fallback(op, idx)
        if prof.enabled:
            prof.substaged.add(idx)
        m, npr = plan.rows, plan.cols // 2
        lines = [
            f"    {{ /* {op.id}: affine layer {idx} (MVP dense, one full-width raw launch -- no scan, no SDK call"
            + ("; FP16 source in place" if plan.fp16_source else f"; FP32 source {plan.source}") + ") */",
            f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
        ]
        if not plan.fp16_source:
            lines += [f"        const float *src = {plan.source};",
                      f"        for (int i = 0; i < {plan.cols}; i++) {{ g_in_f16[{idx}][i] = (float16_t)src[i]; }}",
                      f"        g_mvp_stage_cast_count[{idx}]++;"]
        lines += [
            *prof.lap(f"affine{idx}.stage", "        "),
            "        iscas_sp_enter();",
            f"        iscas_sp_setup({m}, {npr}, g_cur_f16[{idx}], g_sparse_acc[{idx}]);",
            f"        iscas_sp_launch(g_weight_f16[{idx}], g_in_f16[{idx}], 0, {npr}, 1, 1);",
            "        iscas_sp_exit();",
            f"        g_sparse_launch_count[{idx}] += 1;",
            f"        g_sparse_pair_count[{idx}] += {npr};",
            *prof.lap(f"affine{idx}.mvp", "        "),
            f"        for (int j = 0; j < {m}; j++) {{",
            f"            g_current[{idx}][j] = (float)g_cur_f16[{idx}][j] + (L->bias ? L->bias[j] : 0.0f);",
            "        }",
            *prof.lap(f"affine{idx}.out", "        "),
            "#ifdef ISCAS_SPARSE_VERIFY",
            "        {   /* correctness build only (never timed): the deployed dense SDK kernel, compared bitwise */",
            f"            sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_sparse_ref[{idx}]);",
            f"            g_sparse_verify_mismatch[{idx}] += memcmp(g_sparse_ref[{idx}], g_cur_f16[{idx}], {m} * sizeof(float16_t)) != 0;",
            "        }",
            "#endif",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
        ]
        return lines

    return emitter


def mvp_dense_affine_emitter_profiled(direct_fused: Set[int], prof: StageProfile):
    """The dense MVP emitters with stage markers between cast, matvec and output conversion."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        prof.substaged.add(idx)
        lines = [f"    {{ /* {op.id}: affine layer {idx} (MVP dense, profiled) */",
                 f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];"]
        if idx not in direct_fused:
            lines += [f"        for (int i = 0; i < L->in_features; i++) {{ g_in_f16[{idx}][i] = (float16_t){buffer_ref(op.input)}[i]; }}",
                      f"        g_mvp_stage_cast_count[{idx}]++;"]
        lines += [
            *prof.lap(f"affine{idx}.stage", "        "),
            f"        sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_cur_f16[{idx}]);",
            *prof.lap(f"affine{idx}.mvp", "        "),
            "        for (int j = 0; j < L->out_features; j++) {",
            f"            g_current[{idx}][j] = (float)g_cur_f16[{idx}][j] + (L->bias ? L->bias[j] : 0.0f);",
            "        }",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
            *prof.lap(f"affine{idx}.out"),
        ]
        return lines

    return emitter


# Dense input-layer variants:
#   dense_lean      the FP32->FP16 cast reads the step input directly (no identity copy), then the
#                   SDK matvec.
#   dense_window=W  buffers W steps of FP16 input and runs one raw MVP program over the window: an
#                   outer step loop around the SDK matvec's own per-row sequence (CLEAR; MACR2A; ADDR
#                   and STORE), so every output is bitwise equal to the SDK call while the launch cost
#                   is paid once per window. The rest of the network then replays the W steps in
#                   order. This delays hidden-layer updates by up to W-1 steps, so it is not causal
#                   per step. (sl_math_mvp_matrix_mult_f16 is not used: it accumulates in one FP16
#                   lane, a different rounding sequence.)


@dataclass(frozen=True)
class DenseInputPlan:
    layer: int
    rows: int
    cols: int
    absorbed_convert: str
    window: int              # 1 = unbatched lean dense


def dense_input_plan(module: ExecutionModule, window: int = 1) -> Optional[DenseInputPlan]:
    """The loop-body input affine fed by a convert op of the external input that has no other reader."""
    ops_by_id = {op.id: op for op in module.walk_ops()}
    prod, cons = producers(module), consumers(module)
    loop_ids = {op.id for op in module.loop[0].body}
    for op in module.loop[0].body:
        if not isinstance(op, AffineOp):
            continue
        conv = ops_by_id.get(prod.get(op.input))
        if (isinstance(conv, ConvertOp) and conv.id in loop_ids and conv.input == module.input_buffer
                and cons.get(op.input, []) == [op.id]):
            rows, cols = op.weight_layer.out_features, op.weight_layer.in_features
            if window > 1:
                if module.num_timesteps % window or cols % 2 or window > EFR32_MVP_MAX_DIM:
                    raise ValueError(f"dense_window={window} needs an even input width and T % W == 0")
            return DenseInputPlan(layer=layer_index_of(op.output), rows=rows, cols=cols,
                                  absorbed_convert=conv.id, window=window)
    return None


DENSE_WINDOW_HELPERS = """
/* ---- windowed dense MVP realization of the input layer ---- */
#define ISCAS_WN_T16   ((uint32_t)SLI_MVP_DATATYPE_BINARY16 << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT)
#define ISCAS_WN_T16C  ((uint32_t)SLI_MVP_DATATYPE_COMPLEX_BINARY16 << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT)
#define ISCAS_WN_DIM(size, stride) ((((uint32_t)(size) - 1u) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | ((uint32_t)(stride) << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT))
/* Z[w][r] = sum over pairs of W[r][p] * X[w][p] for w < nwin -- per (w, r) exactly the SDK matvec's
 * CLEAR; MACR2A x npairs; ADDR + STORE. Loops: 0 = timestep (outer), 1 = row, 2 = pair (inner). */
static void iscas_wn_matvec_batch(const float16_t *W, const float16_t *X, float16_t *Z, int rows, int npairs, int nwin)
{
    sli_mvp_cmd_enable();
    MVP->ARRAY[0].DIM0CFG = ISCAS_WN_T16C;
    MVP->ARRAY[0].DIM1CFG = ISCAS_WN_DIM(rows, npairs);
    MVP->ARRAY[0].DIM2CFG = ISCAS_WN_DIM(npairs, 1);
    MVP->ARRAY[0].ADDRCFG = (uint32_t)W;
    MVP->ARRAY[1].DIM0CFG = ISCAS_WN_T16C;
    MVP->ARRAY[1].DIM1CFG = ISCAS_WN_DIM(nwin, npairs);
    MVP->ARRAY[1].DIM2CFG = ISCAS_WN_DIM(npairs, 1);
    MVP->ARRAY[1].ADDRCFG = (uint32_t)X;
    MVP->ARRAY[2].DIM0CFG = ISCAS_WN_T16;
    MVP->ARRAY[2].DIM1CFG = ISCAS_WN_DIM(nwin, rows);
    MVP->ARRAY[2].DIM2CFG = ISCAS_WN_DIM(rows, 1);
    MVP->ARRAY[2].ADDRCFG = (uint32_t)Z;
    MVP->LOOP[1].RST = 0;
    MVP->LOOP[2].RST = 0;
    MVP->LOOP[0].CFG = ((uint32_t)(nwin - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT);
    MVP->LOOP[1].CFG = ((uint32_t)(rows - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                       | ((SLI_MVP_LOOP_INCRDIM(SLI_MVP_ARRAY(1), SLI_MVP_INCRDIM_ROW) | SLI_MVP_LOOP_INCRDIM(SLI_MVP_ARRAY(2), SLI_MVP_INCRDIM_ROW)) << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);
    MVP->LOOP[2].CFG = ((uint32_t)(npairs - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                       | (SLI_MVP_LOOP_INCRDIM(SLI_MVP_ARRAY(0), SLI_MVP_INCRDIM_ROW) << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);
    MVP->INSTR[0].CFG0 = SLI_MVP_ALU_Z(SLI_MVP_R3);
    MVP->INSTR[0].CFG1 = 0;
    MVP->INSTR[0].CFG2 = (SLI_MVP_OP(CLEAR) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP0BEGIN | MVP_INSTRCFG2_LOOP1BEGIN;
    MVP->INSTR[1].CFG0 = SLI_MVP_ALU_X(SLI_MVP_R1) | SLI_MVP_ALU_Y(SLI_MVP_R2) | SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3);
    MVP->INSTR[1].CFG1 = SLI_MVP_LOAD(0, SLI_MVP_R1, SLI_MVP_ARRAY(0), SLI_MVP_INCRDIM_COL) | SLI_MVP_LOAD(1, SLI_MVP_R2, SLI_MVP_ARRAY(1), SLI_MVP_INCRDIM_COL);
    MVP->INSTR[1].CFG2 = (SLI_MVP_OP(MACR2A) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP2BEGIN | MVP_INSTRCFG2_LOOP2END;
    MVP->INSTR[2].CFG0 = SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3);
    MVP->INSTR[2].CFG1 = SLI_MVP_STORE(SLI_MVP_R3, SLI_MVP_ARRAY(2), SLI_MVP_INCRDIM_COL);
    MVP->INSTR[2].CFG2 = (SLI_MVP_OP(ADDR) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP0END | MVP_INSTRCFG2_LOOP1END | MVP_INSTRCFG2_ENDPROG;
    MVP->CMD = MVP_CMD_INIT | MVP_CMD_START;
    sli_mvp_cmd_wait_for_completion();
}
"""


def mvp_dense_input_emitter(plan: DenseInputPlan, fallback, prof: StageProfile = NO_PROFILE):
    """Input-layer emitter for ``dense_lean`` / ``dense_window``. Windowed: the batched matvec
    already produced this step's output row ``g_wn_z + s*rows`` (``s`` is the replay index)."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        if idx != plan.layer:
            return fallback(op, idx)
        if prof.enabled:
            prof.substaged.add(idx)
        m = plan.rows
        if plan.window > 1:
            lines = [f"    {{ /* {op.id}: affine layer {idx} (MVP dense, windowed W={plan.window}: output row s of the batched matvec) */",
                     f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
                     f"        const float16_t *z = g_wn_z + (size_t)s * {m};"]
            src = "z"
        else:
            lines = [f"    {{ /* {op.id}: affine layer {idx} (MVP dense, lean: {plan.absorbed_convert} absorbed, cast from the step input) */",
                     f"        const iscas_layer_t *L = &ISCAS_LAYERS[{idx}];",
                     f"        for (int i = 0; i < {plan.cols}; i++) {{ g_in_f16[{idx}][i] = (float16_t)step_input[i]; }}",
                     f"        g_mvp_stage_cast_count[{idx}]++;",
                     *prof.lap(f"affine{idx}.stage", "        "),
                     f"        sl_math_mvp_matrix_vector_mult_f16(&g_mat[{idx}], g_in_f16[{idx}], g_cur_f16[{idx}]);",
                     *prof.lap(f"affine{idx}.mvp", "        ")]
            src = f"g_cur_f16[{idx}]"
        lines += [
            f"        for (int j = 0; j < {m}; j++) {{",
            f"            g_current[{idx}][j] = (float){src}[j] + (L->bias ? L->bias[j] : 0.0f);",
            "        }",
            f"        g_affine_exec_count[{idx}]++;",
            "    }",
            *prof.lap(f"affine{idx}.out"),
        ]
        return lines

    return emitter


RANGE_COUNTERS = ("launch", "pair", "range", "empty", "fallback")


def range_declarations(plans: Dict[int, SparseMvpPlan]) -> List[str]:
    if not plans:
        return []
    out = [
        '#include "sl_mvp.h"',
        '#include "em_device.h"',
        "static float16_t *g_sparse_acc[ISCAS_NUM_LAYERS];   /* per-row FP16 (even, odd) lane accumulator pairs */",
        "static uint16_t *g_sparse_rs[ISCAS_NUM_LAYERS];     /* merged active pair ranges [rs, re) */",
        "static uint16_t *g_sparse_re[ISCAS_NUM_LAYERS];",
        "#ifdef ISCAS_SPARSE_VERIFY",
        "static float16_t *g_sparse_ref[ISCAS_NUM_LAYERS];",
        "#endif",
    ]
    out += [f"static int g_sparse_{c}_count[ISCAS_NUM_LAYERS];" for c in RANGE_COUNTERS]
    out.append("static int g_sparse_verify_mismatch[ISCAS_NUM_LAYERS];")
    out.append(SPARSE_MVP_HELPERS)
    return out


def range_init_lines(plans: Dict[int, SparseMvpPlan]) -> List[str]:
    out: List[str] = []
    for idx, p in sorted(plans.items()):
        out += [
            f'    g_sparse_acc[{idx}] = (float16_t *)snn_alloc_check(malloc((size_t){2 * p.rows} * sizeof(float16_t)), "g_sparse_acc");',
            f'    g_sparse_rs[{idx}] = (uint16_t *)snn_alloc_check(malloc((size_t){p.cols // 2} * sizeof(uint16_t)), "g_sparse_rs");',
            f'    g_sparse_re[{idx}] = (uint16_t *)snn_alloc_check(malloc((size_t){p.cols // 2} * sizeof(uint16_t)), "g_sparse_re");',
            "#ifdef ISCAS_SPARSE_VERIFY",
            f'    g_sparse_ref[{idx}] = (float16_t *)snn_alloc_check(malloc((size_t){p.rows} * sizeof(float16_t)), "g_sparse_ref");',
            "#endif",
        ]
    return out


def range_reset_lines(plans: Dict[int, SparseMvpPlan]) -> List[str]:
    if not plans:
        return []
    out = [f"    memset(g_sparse_{c}_count, 0, sizeof(g_sparse_{c}_count));" for c in RANGE_COUNTERS]
    out.append("    memset(g_sparse_verify_mismatch, 0, sizeof(g_sparse_verify_mismatch));")
    return out


def range_accessor_lines(plans: Dict[int, SparseMvpPlan]) -> List[str]:
    if not plans:
        return []
    out = [f"int iscas_debug_sparse_{c}_count(int layer) {{ return g_sparse_{c}_count[layer]; }}" for c in RANGE_COUNTERS]
    out += [
        "int iscas_debug_sparse_verify_mismatch(int layer) { return g_sparse_verify_mismatch[layer]; }",
        "int iscas_debug_sparse_mvp_fault(void) { return g_sparse_mvp_fault; }",
        "int iscas_debug_sparse_layer(int layer) { switch (layer) { "
        + " ".join(f"case {i}: return 1;" for i in sorted(plans)) + " default: return 0; } }",
    ]
    return out
