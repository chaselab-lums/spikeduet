"""Experimental CPU/MVP overlap (``async_mvp=True``), plus the ``lif_params_sram`` option.

Off by default; a backend schedule transform, the IR is unchanged.

Blocking schedule of one timestep (the existing realization, unchanged):
    ... upstream ops ... | plan | MVP launches (the Cortex-M33 busy-waits) | FP16->FP32 + bias | consumer LIF
Pipelined schedule (same iscas_infer_step() protocol: one input vector per call, nothing buffered ahead):
    ... upstream ops ... | plan | START the launch chain |
        while it runs: the PREVIOUS timestep's "FP16->FP32 + bias" and consumer LIF (the deferred tail), with a
        completion poll every ASYNC_POLL_NEURONS neurons that starts the next range |
        wait for the chain | swap the two MVP output vectors
    iscas_infer_readout() first runs the last timestep's tail, then the (folded) output affine as before.

Arithmetic: every loop body is the text the blocking realization emits, in the same per-neuron order; only the
moment at which timestep t-1's tail executes changes (it needs nothing computed in timestep t). The generated code is
therefore bit-identical to the blocking build (checked by tests/test_efr32_async.py on the host).

Hazards and how each is removed:
  * the MVP writes timestep t's output vector while the tail reads timestep t-1's   -> two vectors, swapped per step
  * an all-zero timestep launches nothing                                          -> per-vector `empty` flag
  * the spike vector / range lists the chain reads                                 -> written only before START;
                                                                                      the next write is after the wait
  * memory ordering                                                                -> __DMB() before START and after the wait
  * MVP faults                                                                     -> recorded, the chain still terminates
  * iscas_infer_reset() between inferences                                         -> clears the pending tail; the
                                                                                      MVP is always idle outside a step
"""
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional

from spikeduet.backend.schedule_common import layer_index_of
from spikeduet.ir.execution import AffineOp, ExecutionModule, LIFUpdateOp

ASYNC_POLL_NEURONS = 32      # measured: 3.3% loop overhead, ~240 idle MVP cycles per launch (16: 6.7%; 64: starves)


@dataclass(frozen=True)
class AsyncMvpPlan:
    layer: int           # the range-sparse MVP affine that runs asynchronously
    affine_id: str
    lif_id: str          # its consumer LIF, deferred by one timestep
    rows: int
    cols: int
    poll_neurons: int = ASYNC_POLL_NEURONS
    mode: str = "fold"           # "fold": tail = this LIF, then the folded output affine
    #                              "chain": tail = every op after the affine in the timestep + the readout sum
    tail_op_ids: tuple = ()      # chain mode: ids of the deferred downstream ops, in schedule order
    tail_m33_layers: tuple = ()  # chain mode: layer indices of the deferred Cortex-M33 affines
    fast_wait: bool = False      # critical-path option: register-only wait loop (iscas_sp_wait_chain)
    poll_m33: bool = False       # critical-path option: completion polls between the row blocks of deferred M33 kernels


def async_mvp_plan(module: ExecutionModule, sparse_plans: Dict[int, object], fold,
                   m33_layers: FrozenSet[int] = frozenset(), *, poll_neurons: Optional[int] = None,
                   fast_wait: bool = False, poll_m33: bool = False) -> AsyncMvpPlan:
    """Checks legality (see ``_async_mvp_plan``) and applies the options ``poll_neurons`` (completion-poll
    interval inside deferred loops), ``fast_wait`` and ``poll_m33``."""
    if poll_neurons is not None and int(poll_neurons) < 1:
        raise ValueError("async_poll_neurons must be >= 1")
    plan = _async_mvp_plan(module, sparse_plans, fold, m33_layers)
    return replace(plan, poll_neurons=ASYNC_POLL_NEURONS if poll_neurons is None else int(poll_neurons),
                   fast_wait=bool(fast_wait), poll_m33=bool(poll_m33))


def _async_mvp_plan(module: ExecutionModule, sparse_plans: Dict[int, object], fold,
                    m33_layers: FrozenSet[int] = frozenset()) -> AsyncMvpPlan:
    """Legality of step-API pipelining; raises ValueError naming the first violated condition.

    Fold mode: exactly one range-sparse MVP affine with an FP16 source in place; its
    result is consumed only by one non-inhibited LIF; everything after that LIF in the timestep is the folded output
    affine (so the deferred tail is the last work of the step and iscas_infer_readout() can flush it)."""
    body = module.loop[0].body
    if len(sparse_plans) != 1:
        raise ValueError(f"async_mvp: needs exactly one range-sparse MVP layer, found {sorted(sparse_plans)}")
    (layer, sp), = sparse_plans.items()
    aff = next((op for op in body if isinstance(op, AffineOp) and layer_index_of(op.output) == layer), None)
    if aff is None:
        raise ValueError("async_mvp: the range-sparse affine is not in the temporal loop")
    if fold is None:
        # chain mode: the whole rest of the timestep is deferred. It must be Cortex-M33-only work (a second MVP layer
        # would need the accelerator while the chain owns it) and must not read anything the next step overwrites
        # before the tail runs: by construction it reads only the MVP output vector (double-buffered) and state.
        tail = body[body.index(aff) + 1:]
        for op in tail:
            if isinstance(op, AffineOp) and layer_index_of(op.output) not in m33_layers:
                raise ValueError(f"async_mvp: downstream affine {op.id} is not a Cortex-M33 layer (cpu_sparse_layers); "
                                 "a second MVP layer cannot run while the launch chain owns the accelerator")
            if isinstance(op, LIFUpdateOp) and op.semantics.inhibition:
                raise ValueError(f"async_mvp: deferred LIF {op.id} uses inhibition (layer-wide update; not chunked)")
        if not any(isinstance(op, LIFUpdateOp) and op.current == aff.output for op in tail):
            raise ValueError("async_mvp: the asynchronous affine must feed a LIF in the same timestep")
        return AsyncMvpPlan(layer=layer, affine_id=aff.id, lif_id="", rows=sp.rows, cols=sp.cols, mode="chain",
                            tail_op_ids=tuple(op.id for op in tail),
                            tail_m33_layers=tuple(layer_index_of(op.output) for op in tail if isinstance(op, AffineOp)))
    if not sp.fp16_source:
        raise ValueError("async_mvp: the range-sparse layer must read an FP16 spike vector in place (a fused hidden edge)")
    lifs = [op for op in body if isinstance(op, LIFUpdateOp) and op.current == aff.output]
    if len(lifs) != 1:
        raise ValueError("async_mvp: the asynchronous affine must feed exactly one LIF")
    lif = lifs[0]
    if lif.semantics.inhibition:
        raise ValueError("async_mvp: the deferred LIF uses inhibition (layer-wide update; not chunked)")
    if fold is None or fold.producer_lif_id != lif.id:
        raise ValueError("async_mvp: the deferred LIF must be the producer of a folded output affine "
                         "(fold_output_affine=True); other downstream work is not deferred in this version")
    return AsyncMvpPlan(layer=layer, affine_id=aff.id, lif_id=lif.id, rows=sp.rows, cols=sp.cols)


def _bias_expr(layer: int, sram: bool) -> str:
    return (f"(g_lifp_bias[{layer}] ? g_lifp_bias[{layer}][j] : 0.0f)" if sram else "(L->bias ? L->bias[j] : 0.0f)")


def _chunk_lif(lines: List[str], poll: int) -> List[str]:
    """The emitted LIF block with its per-neuron loop cut into `poll`-neuron chunks and a completion poll before each
    chunk. The loop body is not touched."""
    head = "            for (int j = 0; j < L->out_features; j++) {"
    assert lines.count(head) == 1, "LIF emitter changed: update _chunk_lif"
    i = lines.index(head)
    k = next(q for q in range(i + 1, len(lines)) if lines[q] == "            }")
    return (lines[:i]
            + [f"            for (int j0 = 0; j0 < L->out_features; j0 += {poll}) {{",
               "            iscas_sp_poll();   /* async_mvp: start the next range if the launch in flight is done */",
               f"            const int j1 = (j0 + {poll} < L->out_features) ? j0 + {poll} : L->out_features;",
               "            for (int j = j0; j < j1; j++) {"]
            + lines[i + 1:k + 1] + ["            }"] + lines[k + 1:])


def tail_lines(plan: AsyncMvpPlan, lif_lines: List[str], prof, sram_params: bool, indent: str) -> List[str]:
    """The deferred tail of the previous timestep: FP16->FP32 + bias (as the blocking realization), then the LIF."""
    l, m, p = plan.layer, plan.rows, plan.poll_neurons
    bias = _bias_expr(l, sram_params)
    out = [
        f"if (g_async_pending[{l}]) {{   /* async_mvp: deferred tail of the previous timestep (layer {l}) */",
        f"    const iscas_layer_t *L = &ISCAS_LAYERS[{l}];",
        f"    const float16_t *pc = g_async_prev[{l}];",
        f"    if (g_async_prev_empty[{l}]) {{",
        f"        for (int j = 0; j < {m}; j++) {{ g_current[{l}][j] = 0.0f + {bias}; }}",
        "    } else {",
        f"        for (int j0 = 0; j0 < {m}; j0 += {p}) {{",
        "            iscas_sp_poll();",
        f"            const int j1 = (j0 + {p} < {m}) ? j0 + {p} : {m};",
        f"            for (int j = j0; j < j1; j++) {{ g_current[{l}][j] = (float)pc[j] + {bias}; }}",
        "        }",
        "    }",
        "    (void)L;",
        *prof.lap(f"affine{l}.out", "    "),
        *_chunk_lif(lif_lines, p),
        *prof.lap(f"lif{l}", "    "),
        f"    g_async_pending[{l}] = 0;",
        "}",
    ]
    return [indent + x if x and not x.startswith("#") else x for x in out]


def async_affine_emitter(plan: AsyncMvpPlan, blocking, lif_op: LIFUpdateOp, lif_emitter, prof, sram_params: bool):
    """AffineEmitter: the async layer is re-assembled from the BLOCKING emitter's own text (plan scan, runtime guard,
    verify block) around the start / deferred tail / wait sequence; every other layer delegates unchanged."""

    def emitter(op: AffineOp, idx: int) -> List[str]:
        if idx != plan.layer:
            return blocking(op, idx)
        b = blocking(op, idx)
        i_empty = b.index("        if (nr == 0) {")
        i_else = b.index("        } else {")
        i_enter = b.index("            iscas_sp_enter();")
        i_ver = b.index("#ifdef ISCAS_SPARSE_VERIFY")
        i_end = b.index("#endif", i_ver)
        assert i_empty < i_else < i_enter < i_ver, "blocking range-sparse emitter changed: update async_affine_emitter"
        head = [b[0].replace("(MVP, range-merged sparse", "(MVP, range-merged sparse, ASYNC step-API pipelining"), *b[1:i_empty]]
        guard = b[i_else + 1:i_enter]           # runtime guard (+ the "stage" profile lap)
        verify = b[i_ver:i_end + 1]
        return _async_block(plan, head, guard, verify, prof,
                            tail_lines(plan, lif_emitter(lif_op), prof, sram_params, "        "))

    return emitter


TAIL_MARK, END_MARK = "@@ASYNC_TAIL@@", "@@ASYNC_END@@"


def _async_block(plan: AsyncMvpPlan, head: List[str], guard: List[str], verify: List[str], prof, middle: List[str]) -> List[str]:
    l, m, npr = plan.layer, plan.rows, plan.cols // 2
    if True:
        return [
            *head,
            "        (void)L;",
            "        int started = 0;",
            "        if (nr == 0) {",
            f"            g_sparse_empty_count[{l}]++;",
            "        } else {",
            *guard,
            "            iscas_sp_enter();",
            f"            iscas_sp_setup({m}, {npr}, g_cur_f16[{l}], g_sparse_acc[{l}]);",
            f"            iscas_sp_begin_chain(g_weight_f16[{l}], g_in_f16[{l}], rs, re, nr);",
            "            started = 1;",
            f"            g_sparse_launch_count[{l}] += nr;",
            f"            g_sparse_pair_count[{l}] += span;",
            "        }",
            *prof.lap(f"affine{l}.start", "        "),
            "#ifdef ISCAS_ASYNC_NO_OVERLAP",
            "        if (started) { iscas_sp_wait_chain(); }   /* control build: same code, the tail runs after the MVP */",
            *prof.lap(f"affine{l}.mvp", "        "),
            "#endif",
            *middle,
            "        if (started) { iscas_sp_wait_chain(); iscas_sp_exit(); }",
            *prof.lap(f"affine{l}.wait", "        "),
            *verify,
            f"        g_async_prev_empty[{l}] = (nr == 0);",
            f"        {{ float16_t *t_ = g_async_prev[{l}]; g_async_prev[{l}] = g_cur_f16[{l}]; g_cur_f16[{l}] = t_; }}   /* swap the two output vectors */",
            f"        g_async_pending[{l}] = 1;",
            f"        g_affine_exec_count[{l}]++;",
            "    }",
        ]


# ---------------------------------------------------------------------------
# chain mode: the asynchronous affine is the input layer and the deferred tail is the rest of the
# timestep -- its output conversion, every downstream LIF and Cortex-M33 affine, and the readout sum. The emitters
# leave two markers; chain_rearranger() then moves the downstream ops' own emitted lines into the tail.
# ---------------------------------------------------------------------------
def _slices(b: List[str]):
    i_empty, i_else = b.index("        if (nr == 0) {"), b.index("        } else {")
    i_enter, i_ver = b.index("            iscas_sp_enter();"), b.index("#ifdef ISCAS_SPARSE_VERIFY")
    i_end = b.index("#endif", i_ver)
    assert i_empty < i_else < i_enter < i_ver, "blocking range-sparse emitter changed: update async_mvp"
    head = [b[0].replace("(MVP, range-merged sparse", "(MVP, range-merged sparse, ASYNC step-API pipelining"), *b[1:i_empty]]
    return head, b[i_else + 1:i_enter], b[i_ver:i_end + 1]


def chain_affine_emitter(plan: AsyncMvpPlan, blocking, prof):
    def emitter(op: AffineOp, idx: int) -> List[str]:
        if idx == plan.layer:
            head, guard, verify = _slices(blocking(op, idx))
            return [*_async_block(plan, head, guard, verify, prof, [TAIL_MARK]), END_MARK]
        if op.id in plan.tail_op_ids:     # a deferred Cortex-M33 affine: keep the MVP chain moving around it
            return ["    iscas_sp_poll();", *blocking(op, idx), "    iscas_sp_poll();"]
        return blocking(op, idx)

    return emitter


def chain_lif_emitter(plan: AsyncMvpPlan, inner):
    def emitter(op: LIFUpdateOp) -> List[str]:
        lines = inner(op)
        return _chunk_lif(lines, plan.poll_neurons) if op.id in plan.tail_op_ids else lines

    return emitter


def chain_rearranger(plan: AsyncMvpPlan, prof, sram_params: bool):
    """(loop_lines, readout-sum lines) -> (loop_lines with the tail in place, flush lines for iscas_infer_readout())."""
    l, m, p = plan.layer, plan.rows, plan.poll_neurons
    bias = _bias_expr(l, sram_params)

    def rearrange(loop_lines: List[str], memsum_lines: List[str]):
        i_m, i_e = loop_lines.index(TAIL_MARK), loop_lines.index(END_MARK)
        pre, post, down = loop_lines[:i_m], loop_lines[i_m + 1:i_e], loop_lines[i_e + 1:] + memsum_lines
        tail = [
            f"if (g_async_pending[{l}]) {{   /* async_mvp: deferred tail of the previous timestep (everything after affine{l}) */",
            f"    const iscas_layer_t *L = &ISCAS_LAYERS[{l}];",
            f"    const float16_t *pc = g_async_prev[{l}];",
            f"    if (g_async_prev_empty[{l}]) {{",
            f"        for (int j = 0; j < {m}; j++) {{ g_current[{l}][j] = 0.0f + {bias}; }}",
            "    } else {",
            f"        for (int j0 = 0; j0 < {m}; j0 += {p}) {{",
            "            iscas_sp_poll();",
            f"            const int j1 = (j0 + {p} < {m}) ? j0 + {p} : {m};",
            f"            for (int j = j0; j < j1; j++) {{ g_current[{l}][j] = (float)pc[j] + {bias}; }}",
            "        }",
            "    }",
            "    (void)L;",
            *prof.lap(f"affine{l}.out", "    "),
            *down,
            f"    g_async_pending[{l}] = 0;",
            "}",
        ]
        ind = lambda pad: [pad + x if x and not x.startswith("#") else x for x in tail]
        assert TAIL_MARK not in down and END_MARK not in down
        return ([*pre, *ind("        "), *post],
                ["    /* async_mvp: flush -- the last timestep's deferred tail runs here, before the readout */", *ind("    ")])

    return rearrange


def kernel_poll_lines(plan: Optional[AsyncMvpPlan], lines: List[str]) -> List[str]:
    """`poll_m33`: inside the accumulate kernels of the DEFERRED Cortex-M33 affines, a completion poll between two
    16-row register blocks (no accumulator is live there, so the kernel arithmetic and its register use are untouched).
    Without the option, or for any other layer, the lines are returned unchanged."""
    if plan is None or not plan.poll_m33 or not plan.tail_m33_layers:
        return lines
    heads = tuple(f"ISCAS_CSP_OPT static void iscas_csp_acc{p}{l}(" for l in plan.tail_m33_layers for p in ("", "p"))
    out, inside = [], False
    for i, x in enumerate(lines):
        out.append(x)
        if x.startswith(heads):
            inside = True
        elif inside and x == "}":
            inside = False
        elif inside and x == "    }" and i + 1 < len(lines) and lines[i + 1].startswith("    {   /* rows "):
            out.append("    iscas_sp_poll();   /* async_mvp: keep the MVP launch chain moving between row blocks */")
    return out


def async_lif_emitter(plan: AsyncMvpPlan, inner):
    def emitter(op: LIFUpdateOp) -> List[str]:
        if op.id == plan.lif_id:
            return [f"    /* {op.id}: DEFERRED (async_mvp): executes during the next timestep's MVP chain, or in iscas_infer_readout() */"]
        return inner(op)

    return emitter


def declarations(plan: AsyncMvpPlan, hw_helpers: str, chain_helpers: str, chain_helpers_fast_wait: str = "") -> List[str]:
    return [hw_helpers, chain_helpers_fast_wait if plan.fast_wait else chain_helpers,
            "static float16_t *g_async_prev[ISCAS_NUM_LAYERS];      /* async_mvp: MVP output vector of the previous timestep */",
            "static int g_async_prev_empty[ISCAS_NUM_LAYERS];       /* ... which was an all-zero timestep (no launch, current = bias) */",
            "static int g_async_pending[ISCAS_NUM_LAYERS];          /* ... whose tail (output conversion + LIF) has not run yet */"]


def init_lines(plan: AsyncMvpPlan) -> List[str]:
    return [f'    g_async_prev[{plan.layer}] = (float16_t *)snn_alloc_check(malloc((size_t){plan.rows} * sizeof(float16_t)), "g_async_prev");',
            "    (void)iscas_sp_launch;   /* the blocking helper stays in the file; an async build does not call it */"]


def reset_lines(plan: AsyncMvpPlan) -> List[str]:
    return [f"    g_async_pending[{plan.layer}] = 0; g_async.active = 0;   /* async_mvp: no tail pending, MVP idle */",
            "#ifdef ISCAS_ASYNC_PERF", "    g_async_run = 0; g_async_stall = 0; g_async_launches = 0;", "#endif"]


def accessor_lines(plan: AsyncMvpPlan) -> List[str]:
    return ["#ifdef ISCAS_ASYNC_PERF",
            "unsigned long iscas_debug_async_run(void) { return (unsigned long)g_async_run; }",
            "unsigned long iscas_debug_async_stall(void) { return (unsigned long)g_async_stall; }",
            "unsigned long iscas_debug_async_launches(void) { return (unsigned long)g_async_launches; }",
            "#endif"]


# ---------------------------------------------------------------------------
# lif_params_sram (opt-in, independent of async_mvp, also valid for the Cortex-M33 engine): the listed layers'
# per-neuron beta / threshold (and, for an async layer, the bias of its output-conversion loop) are read from SRAM
# copies made in snn_init() instead of from the const flash arrays of model_data.h. Same values, same arithmetic.
# ---------------------------------------------------------------------------
def lif_params_sram_emitter(layers: FrozenSet[int], inner):
    def emitter(op: LIFUpdateOp) -> List[str]:
        lines = inner(op)
        idx = layer_index_of(op.current)
        if idx not in layers:
            return lines
        out, n = [], 0
        for x in lines:
            y = (x.replace("iscas_param_at(L->beta, ", f"iscas_param_at(g_lifp_beta[{idx}], ")
                  .replace("iscas_param_at(L->threshold, ", f"iscas_param_at(g_lifp_thr[{idx}], "))
            n += y != x
            out.append(y)
        if not op.semantics.inhibition:
            assert n == 2, "LIF emitter changed: update lif_params_sram_emitter"
        return out

    return emitter


def lif_params_declarations(layers: FrozenSet[int]) -> List[str]:
    if not layers:
        return []
    return ["static float *g_lifp_beta[ISCAS_NUM_LAYERS];   /* lif_params_sram: SRAM copies of the flash LIF parameters */",
            "static float *g_lifp_thr[ISCAS_NUM_LAYERS];",
            "static float *g_lifp_bias[ISCAS_NUM_LAYERS];"]


def lif_params_init_lines(layers: FrozenSet[int]) -> List[str]:
    out: List[str] = []
    for l in sorted(layers):
        out += [f"    {{   /* lif_params_sram: layer {l} */",
                f"        const iscas_layer_t *L = &ISCAS_LAYERS[{l}];",
                f'        g_lifp_beta[{l}] = (float *)snn_alloc_check(malloc((size_t)L->beta_len * sizeof(float)), "g_lifp_beta");',
                f"        memcpy(g_lifp_beta[{l}], L->beta, (size_t)L->beta_len * sizeof(float));",
                f'        g_lifp_thr[{l}] = (float *)snn_alloc_check(malloc((size_t)L->threshold_len * sizeof(float)), "g_lifp_thr");',
                f"        memcpy(g_lifp_thr[{l}], L->threshold, (size_t)L->threshold_len * sizeof(float));",
                f"        g_lifp_bias[{l}] = NULL;",
                "        if (L->bias) {",
                f'            g_lifp_bias[{l}] = (float *)snn_alloc_check(malloc((size_t)L->out_features * sizeof(float)), "g_lifp_bias");',
                f"            memcpy(g_lifp_bias[{l}], L->bias, (size_t)L->out_features * sizeof(float));",
                "        }",
                "    }"]
    return out
