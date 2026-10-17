"""EFR32 backend: ``Efr32Backend`` and the generated CPU and MVP engines.

``Efr32Backend.generate(module)`` emits ``model_data.h``. With ``engine="cpu"`` or ``"mvp"`` it also
emits ``snn_generated_<engine>.c``, which implements ``snn_init()``, ``run_snn_inference()`` and the
streaming ``iscas_infer_reset/step/readout()`` that ``runtime/efr32/app.c`` calls. The engine
executes the Execution IR schedule literally (preheader ops once, loop ops every step) and calls
the LIF helpers in ``runtime/efr32/snn.h``. ``engine="auto"`` picks per-layer engines with
``placement.plan_engine_placement``.

Layer realizations live in ``mvp_emitters`` (MVP), ``cpu_emitters`` (event-driven Cortex-M33),
``fold`` (output-affine folding) and ``async_mvp`` (experimental CPU/MVP overlap); cycle models in
``cost_model``.
"""

from dataclasses import replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from spikeduet.backend.base import Backend, GeneratedArtifact
from spikeduet.backend.c_common import render_model_data_header
from spikeduet.backend.efr32 import async_mvp as _async
from spikeduet.backend.efr32.cpu_emitters import (
    check_pinned_columns,
    check_weight_placement,
    cpu_accessor_lines,
    cpu_affine_emitter,
    cpu_declarations,
    cpu_init_lines,
    cpu_reset_lines,
    cpu_sparse_plan,
)
from spikeduet.backend.efr32.fold import (
    OutputFoldPlan,
    fold_affine_emitter,
    fold_lif_emitter,
    fold_readout_lines,
    output_fold_plan,
)
from spikeduet.backend.efr32.mvp_emitters import (
    DENSE_WINDOW_HELPERS,
    DenseInputPlan,
    dense_input_plan,
    mvp_dense_affine_emitter_profiled,
    mvp_dense_input_emitter,
    mvp_direct_affine_emitter,
    mvp_direct_fused_layer_indices,
    mvp_direct_lif_emitter,
    mvp_full_width_affine_emitter,
    mvp_full_width_plans,
    mvp_range_affine_emitter,
    mvp_sparse_plan,
    range_accessor_lines,
    range_declarations,
    range_init_lines,
    range_reset_lines,
)
from spikeduet.backend.efr32.mvp_sparse import (
    ASYNC_MVP_CHAIN_HELPERS,
    ASYNC_MVP_CHAIN_HELPERS_FAST_WAIT,
    ASYNC_MVP_HW_HELPERS,
)
from spikeduet.backend.efr32.profiling import NO_PROFILE, StageProfile
from spikeduet.backend.schedule_common import buffer_ref, emit_lif, emit_schedule, layer_index_of, scalar_affine_emitter
from spikeduet.ir.execution import AffineOp, ExecutionModule, LIFUpdateOp, TemporalMode


def _input_temporal_flag(module: ExecutionModule) -> int:
    """``ISCAS_INPUT_TEMPORAL``: 1 when the input buffer is ``VARYING`` (one vector per step)."""
    input_mode = module.get_buffer(module.input_buffer).temporal
    return 1 if input_mode is TemporalMode.VARYING else 0


def _common_declarations(module: ExecutionModule) -> str:
    n = "ISCAS_NUM_LAYERS"
    return "\n".join(
        [
            f"static float *g_mem[{n}];",
            f"static float *g_stage[{n}];       /* ConvertOp output */",
            f"static float *g_current[{n}];",
            f"static float *g_spike[{n}];",
            f"static float *g_inhib_mem_state[{n}];",
            f"static float *g_inhib_reset_old[{n}];",
            "static float *g_mem_sum;  /* [last layer out_features], debug exposure */",
            f"static int g_affine_exec_count[{n}];",
            f"static int g_lif_exec_count[{n}];",
            f"static int g_convert_exec_count[{n}];",
        ]
    )




def _infer_functions(
    module: ExecutionModule, affine_emitter, *, lif_emitter=None, extra_reset_lines: Optional[List[str]] = None,
    absorbed_op_ids: FrozenSet[str] = frozenset(), prof: StageProfile = NO_PROFILE,
    window: Optional["DenseInputPlan"] = None,
    fold: Optional[OutputFoldPlan] = None, fold_readout_lines: Optional[List[str]] = None,
    async_chain=None,
) -> str:
    """The engine's inference functions.

    ``iscas_infer_reset(first_input)`` zeroes the state and runs the preheader,
    ``iscas_infer_step(step_input)`` runs one loop iteration on one input vector, and
    ``iscas_infer_readout()`` returns the argmax of the summed output membranes.
    ``run_snn_inference(input)`` drives the three over a whole input tensor. Streaming one vector
    per step keeps the input buffer at ``ISCAS_INPUT_DIM`` floats.
    """
    kwargs = {"affine_emitter": affine_emitter}
    if absorbed_op_ids:
        kwargs["absorbed_op_ids"] = absorbed_op_ids
    if lif_emitter is not None:
        kwargs["lif_emitter"] = lif_emitter
    if prof.enabled:
        loop_ids = {op.id for op in module.loop[0].body}   # preheader ops are covered by the "reset" stage
        if fold is not None:
            loop_ids -= {fold.lif_id} if fold.producer_lif_id is None else {fold.lif_id, fold.affine_id}
        kwargs["op_epilogue"] = lambda op: prof.epilogue(op) if op.id in loop_ids else []
    # the loop body is emitted first so a profiled build numbers its stages in schedule order
    _, loop_lines = emit_schedule(module, input_param="step_input", direct_input_source=True, **kwargs)
    preheader_lines, _ = emit_schedule(
        module, input_param="first_input", direct_input_source=True,
        **{k: v for k, v in kwargs.items() if k != "op_epilogue"})
    windowed = window is not None and window.window > 1
    if async_chain is not None:     # async chain mode: downstream ops and the readout sum move into the deferred tail
        assert not windowed and fold is None
        loop_lines, fold_readout_lines = async_chain(
            loop_lines, ["    {", "        const int last = ISCAS_NUM_LAYERS - 1;", "        const int out_dim = ISCAS_LAYERS[last].out_features;",
                         "        for (int j = 0; j < out_dim; j++) {", "            g_mem_sum[j] += g_mem[last][j];", "        }", "    }",
                         *prof.lap("memsum")])
    if prof.enabled and not windowed:
        loop_lines = [*prof.start(), *loop_lines]
    if prof.enabled:
        preheader_lines = [*preheader_lines, *prof.lap("reset")]
    memsum_block = [
        "    {",
        "        const int last = ISCAS_NUM_LAYERS - 1;",
        "        const int out_dim = ISCAS_LAYERS[last].out_features;",
        "        for (int j = 0; j < out_dim; j++) {",
        "            g_mem_sum[j] += g_mem[last][j];",
        "        }",
        "    }",
    ]
    if windowed:
        w, n, m, i0 = window.window, window.cols, window.rows, window.layer
        loop_lines = [
            *prof.start(),
            f"    {{   /* window staging: this timestep's input -> FP16 row g_wn_n of the W={w} input window */",
            f"        float16_t *xr = g_wn_x + (size_t)g_wn_n * {n};",
            f"        for (int i = 0; i < {n}; i++) {{ xr[i] = (float16_t)step_input[i]; }}",
            f"        g_mvp_stage_cast_count[{i0}]++;",
            "    }",
            *prof.lap(f"affine{i0}.stage"),
            f"    if (++g_wn_n < {w}) {{",
            "        return;   /* window not full: the rest of the chain runs when it is */",
            "    }",
            f"    iscas_wn_matvec_batch(g_weight_f16[{i0}], g_wn_x, g_wn_z, {m}, {n // 2}, {w});",
            "    g_wn_n = 0;",
            *prof.lap(f"affine{i0}.mvp"),
            "#ifdef ISCAS_SPARSE_VERIFY",
            "    {   /* correctness build only (never timed): every buffered timestep vs the deployed SDK matvec, bitwise */",
            f"        for (int s = 0; s < {w}; s++) {{",
            f"            sl_math_mvp_matrix_vector_mult_f16(&g_mat[{i0}], g_wn_x + (size_t)s * {n}, g_cur_f16[{i0}]);",
            f"            g_wn_verify_mismatch += memcmp(g_cur_f16[{i0}], g_wn_z + (size_t)s * {m}, {m} * sizeof(float16_t)) != 0;",
            "        }",
            "    }",
            "#endif",
            f"    for (int s = 0; s < {w}; s++) {{   /* replay the buffered timesteps in order */",
            *["    " + l for l in loop_lines],
            *["    " + l for l in memsum_block],
            *["    " + l for l in prof.lap("memsum")],
            "    }",
        ]

    lines = [
        "void iscas_infer_reset(const float *first_input)",
        "{",
        "    const int out_dim = ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features;",
        "    (void)first_input;  /* only read by a hoisted preheader op, if any -- see below */",
        "",
        "    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {",
        "        memset(g_mem[l], 0, (size_t)ISCAS_LAYERS[l].out_features * sizeof(float));",
        "    }",
        "    memset(g_affine_exec_count, 0, sizeof(g_affine_exec_count));",
        "    memset(g_lif_exec_count, 0, sizeof(g_lif_exec_count));",
        "    memset(g_convert_exec_count, 0, sizeof(g_convert_exec_count));",
        *(extra_reset_lines or []),
        *(["    g_wn_n = 0;", "    g_wn_verify_mismatch = 0;"] if windowed else []),
        *prof.reset_lines(),
        "",
        "    memset(g_mem_sum, 0, (size_t)out_dim * sizeof(float));",
        *prof.start(),
        "",
        "    /* Preheader: ops the compiler hoisted out of the timestep loop. They may only read the",
        "     * input when it is invariant, so first_input is the vector every step would see. */",
    ]
    if preheader_lines:
        lines.extend(preheader_lines)
    else:
        lines.append("    /* <empty> -- no op was hoisted into the preheader */")
    lines += [
        "}",
        "",
        "void iscas_infer_step(const float *step_input)",
        "{",
        "    (void)step_input;  /* unused when layer 0's input-reading op was hoisted into the preheader */",
        "",
        "    /* --- One temporal-loop iteration's worth of ops ---------------- */",
    ]
    lines.extend(loop_lines)
    if not windowed and fold is None and async_chain is None:
        lines += ["", *memsum_block, *prof.lap("memsum")]
    lines += [
        "}",
        "",
        "int iscas_infer_readout(void)",
        "{",
        "    const int last = ISCAS_NUM_LAYERS - 1;",
        "    const int out_dim = ISCAS_LAYERS[last].out_features;",
        *prof.start(),
        *(fold_readout_lines or []),
        "",
        "    int predicted_class = 0;",
        "    float best = g_mem_sum[0];",
        "    for (int j = 1; j < out_dim; j++) {",
        "        if (g_mem_sum[j] > best) {",
        "            best = g_mem_sum[j];",
        "            predicted_class = j;",
        "        }",
        "    }",
        *prof.lap("readout"),
        "    return predicted_class;",
        "}",
        "",
        "int run_snn_inference(float *input_data)",
        "{",
                "    iscas_infer_reset(input_data);",
        "    for (int t = 0; t < ISCAS_STEPS; t++) {",
        "        iscas_infer_step(ISCAS_INPUT_TEMPORAL ?",
        "            (input_data + (size_t)t * ISCAS_INPUT_DIM) : input_data);",
        "    }",
        "    return iscas_infer_readout();",
        "}",
    ]
    return "\n".join(lines)


def generate_efr32_cpu_schedule(
    module: ExecutionModule, *, sparse_cpu: bool = False, stage_profile: bool = False,
    cpu_sparse_weight_placement: Optional[Dict[int, str]] = None,
    cpu_sparse_pinned_columns: Optional[Dict[int, Sequence[int]]] = None,
    fold_output_affine: bool = False,
    lif_params_sram: FrozenSet[int] = frozenset(),
) -> str:
    """``snn_generated_cpu.c``: every layer on the Cortex-M33 in FP32.

    Args:
        sparse_cpu: every loop-body affine runs event-driven (``cpu_emitters``).
        stage_profile: emit DWT stage markers (active in ``ISCAS_STAGE_PROFILE`` builds).
        cpu_sparse_weight_placement: ``{layer: "sram" | "flash"}`` for every event-driven layer;
            ``None`` leaves it to the ``ISCAS_CPU_SPARSE_WT_SRAM`` build define.
        cpu_sparse_pinned_columns: ``{layer: columns}`` copied to SRAM for flash-resident layers.
        fold_output_affine: run the output affine once per inference (``fold``).
        lif_params_sram: layers whose LIF parameters are read from SRAM copies.
    """
    prof = StageProfile(stage_profile)
    plans = cpu_sparse_plan(module) if sparse_cpu else {}
    check_weight_placement(plans, cpu_sparse_weight_placement)
    pinned = check_pinned_columns(plans, cpu_sparse_weight_placement, cpu_sparse_pinned_columns)
    affine_emitter = cpu_affine_emitter(plans, scalar_affine_emitter, prof, pinned) if plans else scalar_affine_emitter
    absorbed = frozenset(p.absorbed_convert for p in plans.values() if p.absorbed_convert)
    fold = output_fold_plan(module) if fold_output_affine else None
    fold_kw, fold_decl = {}, []
    if fold is not None:
        acc_ref = "g_fold_acc" if fold.producer_lif_id is None else buffer_ref(fold.acc_buffer)
        fold_kw = {"lif_emitter": fold_lif_emitter(fold, _async.lif_params_sram_emitter(frozenset(lif_params_sram), emit_lif)
                                                    if lif_params_sram else None), "fold": fold,
                   "fold_readout_lines": fold_readout_lines(fold, plans, prof)}
        affine_emitter = fold_affine_emitter(fold, affine_emitter)
        if fold.producer_lif_id is None:
            fold_decl = [f"static float g_fold_acc[{fold.cols}];   /* folded output affine: input summed over time */"]
    if lif_params_sram and fold is None:
        fold_kw = {"lif_emitter": _async.lif_params_sram_emitter(frozenset(lif_params_sram), emit_lif)}
    reset = list(cpu_reset_lines(plans))
    if fold is not None:
        reset.append(f"    memset({acc_ref}, 0, {fold.cols}u * sizeof(float));   /* folded output affine accumulator */")
    infer = _infer_functions(module, affine_emitter, prof=prof, absorbed_op_ids=absorbed,
                             extra_reset_lines=reset or None, **fold_kw)
    lines = [
        "// Generated by SpikeDuet -- do not edit by hand",
        "/* Cortex-M33 engine. The schedule below is the compiled Execution IR; LIF semantics come",
        " * from snn.h. */",
        '#include "model_data.h"',
        '#include "snn.h"',
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        f"#define ISCAS_STEPS {module.num_timesteps}",
        f"#define ISCAS_INPUT_TEMPORAL {_input_temporal_flag(module)}",
        "",
        *(["#include <stdint.h>"] if plans else []),
        *prof.declarations(),
        _common_declarations(module),
        *cpu_declarations(module, plans, cpu_sparse_pinned_columns),
        *fold_decl,
        *_async.lif_params_declarations(frozenset(lif_params_sram)),
        "",
        "void snn_init(void)",
        "{",
        "    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {",
        "        const iscas_layer_t *L = &ISCAS_LAYERS[l];",
        '        g_mem[l] = (float *)snn_alloc_check(calloc((size_t)L->out_features, sizeof(float)), "g_mem[l]");',
        '        g_stage[l] = (float *)snn_alloc_check(malloc((size_t)L->in_features * sizeof(float)), "g_stage[l]");',
        '        g_current[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_current[l]");',
        '        g_spike[l] = (float *)snn_alloc_check(calloc((size_t)L->out_features, sizeof(float)), "g_spike[l]");',
        "        if (L->inhibition) {",
        '            g_inhib_mem_state[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_inhib_mem_state[l]");',
        '            g_inhib_reset_old[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_inhib_reset_old[l]");',
        "        } else {",
        "            g_inhib_mem_state[l] = NULL;",
        "            g_inhib_reset_old[l] = NULL;",
        "        }",
        "    }",
        *cpu_init_lines(plans, cpu_sparse_weight_placement, cpu_sparse_pinned_columns),
        *_async.lif_params_init_lines(frozenset(lif_params_sram)),
        '    g_mem_sum = (float *)snn_alloc_check(',
        "        calloc((size_t)ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features, sizeof(float)), \"g_mem_sum\");",
        "}",
        "",
        infer,
        "",
        "int iscas_debug_affine_exec_count(int layer) { return g_affine_exec_count[layer]; }",
        "int iscas_debug_lif_exec_count(int layer) { return g_lif_exec_count[layer]; }",
        "int iscas_debug_convert_exec_count(int layer) { return g_convert_exec_count[layer]; }",
        *cpu_accessor_lines(plans),
        *prof.accessor_lines(),
        "",
    ]
    return "\n".join(lines)


def generate_efr32_mvp_schedule(
    module: ExecutionModule,
    *,
    sparse_mvp: bool = False,
    sparse_activity: Optional[Dict[int, Tuple[float, float]]] = None,
    force_sparse_mvp_layers: FrozenSet[int] = frozenset(),
    cpu_sparse_layers: FrozenSet[int] = frozenset(),
    stage_profile: bool = False,
    dense_lean: bool = False,
    dense_window: int = 1,
    mvp_full_width_layers: FrozenSet[int] = frozenset(),
    mvp_static_io_layers: FrozenSet[int] = frozenset(),
    sparse_mvp_layers: Optional[FrozenSet[int]] = None,
    cpu_sparse_weight_placement: Optional[Dict[int, str]] = None,
    cpu_sparse_pinned_columns: Optional[Dict[int, Sequence[int]]] = None,
    mvp_flash_weight_layers: FrozenSet[int] = frozenset(),
    fold_output_affine: bool = False,
    async_mvp: bool = False,
    sparse_merge_gap: Optional[Dict[int, int]] = None,
    lif_params_sram: FrozenSet[int] = frozenset(),
    async_poll_neurons: Optional[int] = None,
    async_fast_wait: bool = False,
    async_poll_m33: bool = False,
) -> str:
    """``snn_generated_mvp.c``: affine layers on the MVP (FP16), LIF updates on the Cortex-M33.

    A hoisted affine gets exactly one MVP call before the loop. When fusion ran, a non-inhibited
    LIF writes its spikes straight into the next layer's FP16 input, so that edge has no staging
    loop. ``Efr32Backend.generate`` documents every option; all are off by default.
    """
    prof = StageProfile(stage_profile)
    cpu_plans = cpu_sparse_plan(module, set(cpu_sparse_layers)) if cpu_sparse_layers else {}
    check_weight_placement(cpu_plans, cpu_sparse_weight_placement)
    cpu_pinned = check_pinned_columns(cpu_plans, cpu_sparse_weight_placement, cpu_sparse_pinned_columns)
    direct_fused = mvp_direct_fused_layer_indices(module) - set(cpu_plans)
    affine_emitter = (mvp_dense_affine_emitter_profiled(direct_fused, prof) if stage_profile
                      else mvp_direct_affine_emitter(direct_fused))
    lif_emitter = mvp_direct_lif_emitter(direct_fused)
    # Opt-in range-sparse realization (default off: generated text byte-identical to before).
    plans = {}
    if sparse_mvp:
        plans = {i: p for i, p in mvp_sparse_plan(module, activity=sparse_activity,
                                                  force_layers=force_sparse_mvp_layers).items()
                 if i not in cpu_plans and (not p.fp16_source or i in direct_fused)
                 and (sparse_mvp_layers is None or i in sparse_mvp_layers)}
    if sparse_merge_gap:
        if not set(sparse_merge_gap) <= set(plans):
            raise ValueError(f"sparse_merge_gap names layers {sorted(sparse_merge_gap)} that are not range-sparse ({sorted(plans)})")
        plans = {i: (replace(p, gap_pairs=int(sparse_merge_gap[i])) if i in sparse_merge_gap else p) for i, p in plans.items()}
    lif_params_sram = frozenset(lif_params_sram)
    if lif_params_sram:
        lif_emitter = _async.lif_params_sram_emitter(lif_params_sram, lif_emitter)
    if plans:
        affine_emitter = mvp_range_affine_emitter(plans, affine_emitter, prof)
    if cpu_plans:
        affine_emitter = cpu_affine_emitter(cpu_plans, affine_emitter, prof, cpu_pinned)
    fw = mvp_full_width_plans(module, frozenset(mvp_full_width_layers)) if mvp_full_width_layers else {}
    if set(fw) & (set(plans) | set(cpu_plans)):
        raise ValueError("mvp_full_width_layers overlaps another realization of the same layer")
    if fw:
        affine_emitter = mvp_full_width_affine_emitter(fw, affine_emitter, prof)
    fold = output_fold_plan(module) if fold_output_affine else None
    fold_kw, fold_decl, fold_reset = {}, [], []
    if fold is not None:
        if fold.layer not in cpu_plans:
            raise ValueError("fold_output_affine with engine='mvp' needs the output affine on the Cortex-M33 (cpu_sparse_layers)")
        acc_ref = "g_fold_acc" if fold.producer_lif_id is None else buffer_ref(fold.acc_buffer)
        lif_emitter = fold_lif_emitter(fold, lif_emitter)
        affine_emitter = fold_affine_emitter(fold, affine_emitter)
        fold_kw = {"fold": fold, "fold_readout_lines": fold_readout_lines(fold, cpu_plans, prof)}
        if fold.producer_lif_id is None:
            fold_decl = [f"static float g_fold_acc[{fold.cols}];   /* folded output affine: input summed over time */"]
        fold_reset = [f"    memset({acc_ref}, 0, {fold.cols}u * sizeof(float));   /* folded output affine accumulator */"]
    aplan = None
    if (async_poll_neurons is not None or async_fast_wait or async_poll_m33) and not async_mvp:
        raise ValueError("async_poll_neurons / async_fast_wait / async_poll_m33 need async_mvp=True")
    if async_mvp:
        aplan = _async.async_mvp_plan(module, plans, fold, frozenset(cpu_plans), poll_neurons=async_poll_neurons,
                                      fast_wait=async_fast_wait, poll_m33=async_poll_m33)
        sram_bias = aplan.layer in lif_params_sram
        if aplan.mode == "chain":       # the rest of the timestep is the deferred tail
            affine_emitter = _async.chain_affine_emitter(aplan, affine_emitter, prof)
            lif_emitter = _async.chain_lif_emitter(aplan, lif_emitter)
            fold_kw["async_chain"] = _async.chain_rearranger(aplan, prof, sram_bias)
        else:
            lif_op = next(op for op in module.loop[0].body if isinstance(op, LIFUpdateOp) and op.id == aplan.lif_id)
            tail_lif = lif_emitter          # the deferred LIF's own emitter chain (SRAM parameters / fold accumulation)
            affine_emitter = _async.async_affine_emitter(aplan, affine_emitter, lif_op, tail_lif, prof, sram_bias)
            fold_kw["fold_readout_lines"] = [
                "    /* async_mvp: flush -- the last timestep's deferred tail runs here, before the folded output affine */",
                *_async.tail_lines(aplan, tail_lif(lif_op), prof, sram_bias, "    "),
                *fold_kw["fold_readout_lines"]]
            lif_emitter = _async.async_lif_emitter(aplan, lif_emitter)
        fold_reset = [*fold_reset, *_async.reset_lines(aplan)]
    dense_in = dense_input_plan(module, dense_window) if (dense_lean or dense_window > 1) else None
    if dense_in is not None:
        if dense_in.layer in plans or dense_in.layer in cpu_plans:
            raise ValueError("dense_lean/dense_window and a sparse realization of the same layer are exclusive")
        affine_emitter = mvp_dense_input_emitter(dense_in, affine_emitter, prof)
    absorbed = frozenset(p.absorbed_convert for p in list(plans.values()) + list(cpu_plans.values()) + list(fw.values())
                         + ([dense_in] if dense_in else []) if p.absorbed_convert)
    if dense_in is not None and dense_in.layer in fw:
        raise ValueError("dense_lean/dense_window and mvp_full_width_layers of the same layer are exclusive")
    raw_plans = {**plans, **fw}     # every layer using the raw MVP kernel (declarations/counters)
    # mvp_static_io_layers: those layers' FP16 input/output vectors live in static .bss instead of the heap
    # slot next to their weights, so the MVP's two operand streams come from different SRAM regions. The heap
    # allocations are kept (unused) so every other buffer address is unchanged.
    io_dims = {layer_index_of(op.output): (op.weight_layer.in_features, op.weight_layer.out_features)
               for op in module.walk_ops() if isinstance(op, AffineOp)}
    static_io_decl = [line for l in sorted(mvp_static_io_layers) for line in (
        f"static float16_t g_io_in{l}[{io_dims[l][0]}] __attribute__((aligned(4)));  /* static I/O */",
        f"static float16_t g_io_cur{l}[{io_dims[l][1]}] __attribute__((aligned(4)));")]
    static_io_init = [f"    g_in_f16[{l}] = g_io_in{l}; g_cur_f16[{l}] = g_io_cur{l};   /* static I/O */"
                      for l in sorted(mvp_static_io_layers)]
    windowed = dense_in is not None and dense_in.window > 1
    infer = _infer_functions(
        module, affine_emitter, lif_emitter=lif_emitter,
        extra_reset_lines=[
            "    memset(g_mvp_stage_cast_count, 0, sizeof(g_mvp_stage_cast_count));",
            "    memset(g_mvp_direct_fp16_write_count, 0, sizeof(g_mvp_direct_fp16_write_count));",
            *range_reset_lines(raw_plans),
            *cpu_reset_lines(cpu_plans),
            *fold_reset,
        ],
        absorbed_op_ids=absorbed, prof=prof, window=dense_in, **fold_kw,
    )
    flash_w = frozenset(mvp_flash_weight_layers)
    flash_decl: List[str] = []
    if flash_w:
        if dense_in is not None or fw or (flash_w & set(cpu_plans)):
            raise ValueError("mvp_flash_weight_layers applies to dense or range-sparse MVP layers (not M33, windowed or raw full-width layers)")
        import numpy as np
        wts = {layer_index_of(op.output): op.weight_layer.weight for op in module.walk_ops() if isinstance(op, AffineOp)}
        assert flash_w <= set(wts), f"mvp_flash_weight_layers {sorted(flash_w)} are not affine layers"
        flash_decl = ["#include <stdint.h>", "#include <stdio.h>   /* W16FLASH check line (ISCAS_SPARSE_VERIFY builds) */"]
        for l in sorted(flash_w):
            w = wts[l].detach().cpu().numpy() if hasattr(wts[l], "detach") else wts[l]
            bits = np.asarray(w, dtype=np.float32).astype(np.float16).view(np.uint16)
            flash_decl.append(f"static const uint16_t iscas_w16_l{l}[{bits.size}] __attribute__((aligned(4))) = {{   /* flash-resident FP16 W */")
            flash_decl += ["    " + ", ".join(f"0x{v:04x}" for v in row) + "," for row in bits]
            flash_decl.append("};")
        flash_decl.append("static const uint16_t *const iscas_w16_flash[ISCAS_NUM_LAYERS] = {"
                          + ", ".join(f"iscas_w16_l{l}" if l in flash_w else "0" for l in range(len(wts))) + "};")
        weight_init = [
            "        if (iscas_w16_flash[l] != 0) {   /* per-layer placement: the MVP reads this layer's FP16 W in place from flash */",
            "            g_weight_f16[l] = (float16_t *)iscas_w16_flash[l];",
            "#ifdef ISCAS_SPARSE_VERIFY",
            "            {   /* correctness build only: the table must equal the ARM (float16_t) cast, bit for bit */",
            "                const size_t nw = (size_t)L->out_features * (size_t)L->in_features;",
            "                unsigned bad = 0;",
            "                for (size_t idx = 0; idx < nw; idx++) {",
            "                    const float16_t h = (float16_t)L->weight[idx];",
            "                    bad += memcmp(&h, &g_weight_f16[l][idx], sizeof(h)) != 0;",
            "                }",
            '                printf("W16FLASH:%d:mismatch=%u%c", l, bad, 10);',
            "            }",
            "#endif",
            "            sl_math_matrix_init_f16(&g_mat[l], L->out_features, L->in_features, g_weight_f16[l]);",
            '            g_in_f16[l] = (float16_t *)snn_alloc_check(malloc((size_t)L->in_features * sizeof(float16_t)), "g_in_f16[l]");',
            '            g_cur_f16[l] = (float16_t *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float16_t)), "g_cur_f16[l]");',
            "        } else {",
        ]
        weight_close = ["        }"]
        ind = "    "
        if cpu_plans:   # hybrid build: M33 layers have no MVP buffers at all
            weight_init = [
                "        if (" + " || ".join(f"l == {i}" for i in sorted(cpu_plans)) + ") {",
                "            g_weight_f16[l] = NULL; g_in_f16[l] = NULL; g_cur_f16[l] = NULL;   /* Cortex-M33 event-driven layer: no MVP buffers */",
                "        } else " + weight_init[0].lstrip(),
                *weight_init[1:],
            ]
    elif cpu_plans:
        weight_init = [
            "        if (" + " || ".join(f"l == {i}" for i in sorted(cpu_plans)) + ") {",
            "            g_weight_f16[l] = NULL; g_in_f16[l] = NULL; g_cur_f16[l] = NULL;   /* Cortex-M33 event-driven layer: no MVP buffers */",
            "        } else {",
        ]
        weight_close = ["        }"]
        ind = "    "
    else:
        weight_init, weight_close, ind = [], [], ""

    lines = [
        "// Generated by SpikeDuet -- do not edit by hand",
        "/* MVP engine. The schedule below is the compiled Execution IR: a hoisted affine has one MVP",
        " * call before the loop, and fused LIFs write FP16 spikes straight into the next layer's MVP",
        " * input. LIF semantics come from snn.h. */",
        '#include "model_data.h"',
        '#include "snn.h"',
        '#include "sl_math_mvp.h"',
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        f"#define ISCAS_STEPS {module.num_timesteps}",
        f"#define ISCAS_INPUT_TEMPORAL {_input_temporal_flag(module)}",
        "",
        "static sl_math_matrix_f16_t g_mat[ISCAS_NUM_LAYERS];",
        "static float16_t *g_weight_f16[ISCAS_NUM_LAYERS];",
        "static float16_t *g_in_f16[ISCAS_NUM_LAYERS];",
        "static float16_t *g_cur_f16[ISCAS_NUM_LAYERS];",
        "static int g_mvp_stage_cast_count[ISCAS_NUM_LAYERS];       /* FP32->FP16 staging loops executed */",
        "static int g_mvp_direct_fp16_write_count[ISCAS_NUM_LAYERS]; /* LIF loops that wrote FP16 spikes directly */",
        *range_declarations(raw_plans),
        *(_async.declarations(aplan, ASYNC_MVP_HW_HELPERS, ASYNC_MVP_CHAIN_HELPERS, ASYNC_MVP_CHAIN_HELPERS_FAST_WAIT) if aplan else []),
        *_async.lif_params_declarations(lif_params_sram),
        *(['#include "sl_mvp.h"', '#include "em_device.h"',
           "static float16_t *g_wn_x;   /* [W][in] FP16 input window */",
           "static float16_t *g_wn_z;   /* [W][out] FP16 batched matvec output */",
           "static int g_wn_n;          /* timesteps buffered in the current window */",
           "static int g_wn_verify_mismatch;   /* ISCAS_SPARSE_VERIFY builds: timesteps differing from the SDK matvec */",
           DENSE_WINDOW_HELPERS] if windowed else []),
        *prof.declarations(),
        *static_io_decl,
        *flash_decl,
        _common_declarations(module),
        *_async.kernel_poll_lines(aplan, cpu_declarations(module, cpu_plans, cpu_sparse_pinned_columns)),
        *fold_decl,
        "",
        "void snn_init(void)",
        "{",
        "    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {",
        "        const iscas_layer_t *L = &ISCAS_LAYERS[l];",
        *weight_init,
        ind + "        const size_t n_weights = (size_t)L->out_features * (size_t)L->in_features;",
        ind + '        g_weight_f16[l] = (float16_t *)snn_alloc_check(malloc(n_weights * sizeof(float16_t)), "g_weight_f16[l]");',
        ind + "        for (size_t idx = 0; idx < n_weights; idx++) {",
        ind + "            g_weight_f16[l][idx] = (float16_t)L->weight[idx];",
        ind + "        }",
        ind + "        sl_math_matrix_init_f16(&g_mat[l], L->out_features, L->in_features, g_weight_f16[l]);",
        ind + '        g_in_f16[l] = (float16_t *)snn_alloc_check(malloc((size_t)L->in_features * sizeof(float16_t)), "g_in_f16[l]");',
        ind + '        g_cur_f16[l] = (float16_t *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float16_t)), "g_cur_f16[l]");',
        *weight_close,
        '        g_mem[l] = (float *)snn_alloc_check(calloc((size_t)L->out_features, sizeof(float)), "g_mem[l]");',
        '        g_stage[l] = (float *)snn_alloc_check(malloc((size_t)L->in_features * sizeof(float)), "g_stage[l]");',
        '        g_current[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_current[l]");',
        '        g_spike[l] = (float *)snn_alloc_check(calloc((size_t)L->out_features, sizeof(float)), "g_spike[l]");',
        "        if (L->inhibition) {",
        '            g_inhib_mem_state[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_inhib_mem_state[l]");',
        '            g_inhib_reset_old[l] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_inhib_reset_old[l]");',
        "        } else {",
        "            g_inhib_mem_state[l] = NULL;",
        "            g_inhib_reset_old[l] = NULL;",
        "        }",
        "    }",
        *range_init_lines(raw_plans),
        *(_async.init_lines(aplan) if aplan else []),
        *_async.lif_params_init_lines(lif_params_sram),
        *cpu_init_lines(cpu_plans, cpu_sparse_weight_placement, cpu_sparse_pinned_columns),
        *([f'    g_wn_x = (float16_t *)snn_alloc_check(malloc((size_t){dense_in.window * dense_in.cols} * sizeof(float16_t)), "g_wn_x");',
           f'    g_wn_z = (float16_t *)snn_alloc_check(malloc((size_t){dense_in.window * dense_in.rows} * sizeof(float16_t)), "g_wn_z");']
          if windowed else []),
        *static_io_init,
        '    g_mem_sum = (float *)snn_alloc_check(',
        "        calloc((size_t)ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features, sizeof(float)), \"g_mem_sum\");",
        "}",
        "",
        infer,
        "",
        "int iscas_debug_affine_exec_count(int layer) { return g_affine_exec_count[layer]; }",
        "int iscas_debug_lif_exec_count(int layer) { return g_lif_exec_count[layer]; }",
        "int iscas_debug_convert_exec_count(int layer) { return g_convert_exec_count[layer]; }",
        "int iscas_debug_mvp_stage_cast_count(int layer) { return g_mvp_stage_cast_count[layer]; }",
        "int iscas_debug_mvp_direct_fp16_write_count(int layer) { return g_mvp_direct_fp16_write_count[layer]; }",
        *range_accessor_lines(raw_plans),
        *(_async.accessor_lines(aplan) if aplan else []),
        *cpu_accessor_lines(cpu_plans),
        *(["int iscas_debug_window_verify_mismatch(void) { return g_wn_verify_mismatch; }",
           f"int iscas_debug_window_size(void) {{ return {dense_in.window}; }}"] if windowed else []),
        *prof.accessor_lines(),
        "",
    ]
    return "\n".join(lines)


class Efr32Backend(Backend):
    """EFR32 code generator.

    ``generate(module, engine=None | "cpu" | "mvp" | "auto", **options)``; see the README for the
    option reference. All options are off by default.
    """

    name = "efr32"

    def generate(
        self, module: ExecutionModule, *, engine: str = None, sparse_mvp: bool = False,
        sparse_cpu: bool = False, cpu_sparse_layers: FrozenSet[int] = frozenset(),
        force_sparse_mvp_layers: FrozenSet[int] = frozenset(),
        sparse_activity: Optional[Dict[int, Tuple[float, float]]] = None,
        stage_profile: bool = False, dense_lean: bool = False, dense_window: int = 1,
        mvp_full_width_layers: FrozenSet[int] = frozenset(), mvp_static_io_layers: FrozenSet[int] = frozenset(),
        sparse_mvp_layers: Optional[FrozenSet[int]] = None,
        activity_profile=None, force_mvp_layers: FrozenSet[int] = frozenset(), placement_platform=None,
        m33_weights: str = "auto", cpu_sparse_weight_placement: Optional[Dict[int, str]] = None,
        cpu_sparse_pinned_columns: Optional[Dict[int, Sequence[int]]] = None, fold_output_affine: bool = False,
        mvp_flash_weight_layers: FrozenSet[int] = frozenset(),
        async_mvp: bool = False, sparse_merge_gap: Optional[Dict[int, int]] = None,
        lif_params_sram: FrozenSet[int] = frozenset(), async_poll_neurons: Optional[int] = None,
        async_fast_wait: bool = False, async_poll_m33: bool = False, **kwargs,
    ) -> GeneratedArtifact:
        if engine == "auto":
            # The plan resolves to explicit options and is generated through the same paths.
            from spikeduet.backend.efr32.placement import plan_engine_placement
            from spikeduet.target.platform import EFR32MG24_MVP

            unsupported = [name for name, used in (
                ("sparse_mvp", sparse_mvp), ("sparse_cpu", sparse_cpu), ("force_sparse_mvp_layers", force_sparse_mvp_layers),
                ("sparse_activity", sparse_activity), ("dense_lean", dense_lean), ("dense_window", dense_window > 1),
                ("mvp_full_width_layers", mvp_full_width_layers), ("sparse_mvp_layers", sparse_mvp_layers is not None),
                ("cpu_sparse_weight_placement", cpu_sparse_weight_placement is not None),
                ("cpu_sparse_pinned_columns", bool(cpu_sparse_pinned_columns)),
                ("mvp_flash_weight_layers", bool(mvp_flash_weight_layers)), ("async_mvp", async_mvp),
                ("sparse_merge_gap", bool(sparse_merge_gap)), ("lif_params_sram", bool(lif_params_sram))) if used]
            if unsupported:
                raise ValueError(f"engine='auto' decides {unsupported} itself; use an explicit engine to set them by hand")
            plan = plan_engine_placement(
                module, activity_profile, force_m33=frozenset(cpu_sparse_layers), force_mvp=frozenset(force_mvp_layers),
                static_io_override=frozenset(mvp_static_io_layers) if mvp_static_io_layers else None,
                platform=placement_platform or EFR32MG24_MVP, m33_weights=m33_weights, fold_output=fold_output_affine)
            artifact = self.generate(module, stage_profile=stage_profile, **plan.generate_kwargs())
            artifact.files["placement_report.json"] = plan.to_json()
            return artifact
        if activity_profile is not None or force_mvp_layers or placement_platform is not None or m33_weights != "auto":
            raise ValueError("activity_profile/force_mvp_layers/placement_platform/m33_weights apply to engine='auto'")

        header = render_model_data_header(
            module,
            guard="MODEL_DATA_H",
            generator_comment="// Generated by SpikeDuet -- do not edit by hand",
        )

        if engine == "cpu":
            files_engine_suffix = "cpu"
        elif engine == "mvp":
            files_engine_suffix = "mvp"
        elif engine is not None:
            raise ValueError(f"unknown engine {engine!r}; expected 'cpu', 'mvp', 'auto', or None")
        else:
            files_engine_suffix = None
        if sparse_mvp and engine != "mvp":
            raise ValueError("sparse_mvp=True requires engine='mvp' (range-sparse is an MVP realization)")
        if (cpu_sparse_weight_placement is not None or cpu_sparse_pinned_columns) and engine not in ("cpu", "mvp"):
            raise ValueError("cpu_sparse_weight_placement applies to engine='cpu'/'mvp' builds with event-driven layers")
        if mvp_flash_weight_layers and engine != "mvp":
            raise ValueError("mvp_flash_weight_layers applies to engine='mvp' builds")
        if fold_output_affine and engine not in ("cpu", "mvp"):
            raise ValueError("fold_output_affine is implemented for engine='cpu' and hybrid engine='mvp' builds (the MLP generator has its own option)")
        if (async_mvp or sparse_merge_gap or async_poll_neurons is not None or async_fast_wait or async_poll_m33) and engine != "mvp":
            raise ValueError("async_mvp/sparse_merge_gap apply to engine='mvp' builds with a range-sparse layer")
        if lif_params_sram and engine not in ("cpu", "mvp"):
            raise ValueError("lif_params_sram applies to engine='cpu'/'mvp' builds")
        if sparse_cpu and engine != "cpu":
            raise ValueError("sparse_cpu=True requires engine='cpu' (use cpu_sparse_layers= for a hybrid MVP build)")
        if (cpu_sparse_layers or force_sparse_mvp_layers or sparse_activity or dense_lean or dense_window > 1
                or sparse_mvp_layers is not None) and engine != "mvp":
            raise ValueError("cpu_sparse_layers/force_sparse_mvp_layers/sparse_activity/sparse_mvp_layers apply to engine='mvp'")

        if files_engine_suffix is not None:
            # app.c sizes its input buffer from these, so engine builds add them to model_data.h.
            guard_line = "#define MODEL_DATA_H"
            extra = (
                f"{guard_line}\n"
                f"#define ISCAS_STEPS {module.num_timesteps}\n"
                f"#define ISCAS_INPUT_TEMPORAL {_input_temporal_flag(module)}"
            )
            assert guard_line in header
            header = header.replace(guard_line, extra, 1)

        files = {"model_data.h": header}

        if files_engine_suffix is not None:
            files[f"snn_generated_{files_engine_suffix}.c"] = (
                generate_efr32_cpu_schedule(module, sparse_cpu=sparse_cpu, stage_profile=stage_profile,
                                            cpu_sparse_weight_placement=cpu_sparse_weight_placement,
                                            cpu_sparse_pinned_columns=cpu_sparse_pinned_columns,
                                            fold_output_affine=fold_output_affine,
                                            lif_params_sram=frozenset(lif_params_sram))
                if files_engine_suffix == "cpu"
                else generate_efr32_mvp_schedule(
                    module, sparse_mvp=sparse_mvp, sparse_activity=sparse_activity,
                    force_sparse_mvp_layers=frozenset(force_sparse_mvp_layers),
                    cpu_sparse_layers=frozenset(cpu_sparse_layers), stage_profile=stage_profile,
                    dense_lean=dense_lean, dense_window=dense_window,
                    mvp_full_width_layers=frozenset(mvp_full_width_layers),
                    mvp_static_io_layers=frozenset(mvp_static_io_layers),
                    sparse_mvp_layers=None if sparse_mvp_layers is None else frozenset(sparse_mvp_layers),
                    cpu_sparse_weight_placement=cpu_sparse_weight_placement,
                    cpu_sparse_pinned_columns=cpu_sparse_pinned_columns,
                    mvp_flash_weight_layers=frozenset(mvp_flash_weight_layers),
                    fold_output_affine=fold_output_affine, async_mvp=async_mvp,
                    sparse_merge_gap=sparse_merge_gap, lif_params_sram=frozenset(lif_params_sram),
                    async_poll_neurons=async_poll_neurons, async_fast_wait=async_fast_wait, async_poll_m33=async_poll_m33)
            )

        return GeneratedArtifact(files=files)
