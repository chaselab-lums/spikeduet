"""Overlap-aware cost model for ``async_mvp`` builds (experimental; does not change generated code).

``placement.plan_engine_placement`` prices each affine in isolation and sums the costs, i.e. it assumes
the Cortex-M33 and the MVP never run at the same time. With ``async_mvp`` a timestep costs

    step = pre + max(chain, tail)

    pre   : everything before the asynchronous MVP layer k, layer k's range scan and staging, and
            starting the launches
    chain : layer k's launch chain, including the hand-over gap between two polled launches
    tail  : the deferred M33 work that runs while the chain is busy (layer k's output loop of the
            previous step, every later LIF and M33 affine, the readout sum)

A mapping is legal when exactly one loop-resident affine is range-sparse on the MVP and every other
one is event-driven on the M33 (as ``async_mvp.async_mvp_plan`` requires). The per-layer costs come
from ``cost_model`` and ``cpu_sparse``; the async constants below were calibrated on stage profiles
and MVP counters of two measured builds (one in fold mode with flash FP16 weights, one in chain mode
with SRAM weights). Per-step activity (``StepActivity`` arrays) gives E[max(chain, tail)]; mean
activity gives max(E[chain], E[tail]), which is optimistic when the two are close.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np

from spikeduet.backend.efr32.cost_model import (
    EFR32_MVP_FLASH_LAUNCH_FACTOR,
    EFR32_MVP_FLASH_PAIR_FACTOR,
    EFR32_MVP_FP16_SPIKE_WRITE_CYCLES,
    EFR32_MVP_HIDDEN_OUT_CYCLES_PER_ROW,
    mvp_launch_cycles,
    mvp_sparse_affine_cycles,
    mvp_sparse_hidden_affine_cycles,
)
from spikeduet.backend.efr32.cpu_sparse import cpu_sparse_affine_cycles

# Per-neuron M33 costs (measured stage profiles): hidden LIF 24.0-29.7 cycles per neuron-step over four
# layer shapes, no-reset output LIF 20.8-21.3, readout sum 8.7-9.8 per output neuron, input-layer MVP
# output loop 12.2-14.1 per row.
EFR32_LIF_CYCLES_PER_NEURON = 27.5
EFR32_LIF_NORESET_CYCLES_PER_NEURON = 21.0
EFR32_MEMSUM_CYCLES_PER_NEURON = 9.0
EFR32_MVP_INPUT_OUT_CYCLES_PER_ROW = 13.0

# Async-only constants (cycles).
EFR32_ASYNC_START_CYCLES_PER_LAUNCH = 55.0     # start stage: 50-56 per launch
EFR32_ASYNC_GAP_CYCLES_PER_LAUNCH = 355.0      # polled hand-over between two launches (bench: 343-368 at 16-32
#                                                neurons per poll); reproduces both builds' wait stage within ~0.8 ms
EFR32_ASYNC_CHAIN_FACTOR_SRAM = 0.90           # MVP RUN counter / modelled blocking chain, SRAM FP16 weights
EFR32_ASYNC_CHAIN_FACTOR_FLASH = 1.075         # same, flash FP16 weights (flash-port contention)
EFR32_ASYNC_TAIL_OUT_FACTOR = 1.35             # deferred, chunked, polled loops vs their blocking form:
EFR32_ASYNC_TAIL_LIF_FACTOR = 1.22             #   output loop +24..45%, LIF +14..25%,
EFR32_ASYNC_TAIL_M33_AFFINE_FACTOR = 1.10      #   M33 affine +10%, readout sum +3%
EFR32_ASYNC_TAIL_MEMSUM_FACTOR = 1.03


@dataclass(frozen=True)
class AsyncLayer:
    """One loop-resident affine layer + its LIF, as the overlap model needs it."""
    rows: int
    cols: int
    lif_reset: bool = True                 # False: reset_mechanism "none" (cheaper update)
    weights_in_sram: bool = True           # M33: transposed FP32 W; MVP: FP16 W
    hidden: bool = True                    # input is LIF spikes (False: external input, FP32-staged on the MVP)
    folded: bool = False                   # output affine folded into the readout (runs once per inference)


@dataclass(frozen=True)
class StepActivity:
    """Per-timestep input activity of one affine layer (arrays of equal length; length 1 = mean activity)."""
    active: np.ndarray                     # non-zero inputs
    launches: Optional[np.ndarray] = None  # merged MVP ranges (needed for the MVP layer only)
    pairs: Optional[np.ndarray] = None     # processed FP16 column pairs

    @staticmethod
    def of_means(active: float, nonempty_fraction: float = 1.0, launches: Optional[float] = None,
                 pairs: Optional[float] = None) -> "StepActivity":
        a = lambda v: None if v is None else np.array([float(v)])
        s = StepActivity(a(active), a(launches), a(pairs))
        object.__setattr__(s, "_nonempty", float(nonempty_fraction))
        return s

    def nonempty(self) -> np.ndarray:
        return np.array([getattr(self, "_nonempty")]) if hasattr(self, "_nonempty") else (self.active > 0).astype(float)


@dataclass
class AsyncCost:
    mvp_layer: Optional[int]               # None: the all-M33 mapping
    blocking_cycles: float                 # mean per timestep, plain sum (what the per-layer planner assumes)
    async_cycles: Optional[float]          # mean per timestep under step-API pipelining
    parts: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _lif(layer: AsyncLayer) -> float:
    return layer.rows * (EFR32_LIF_CYCLES_PER_NEURON if layer.lif_reset else EFR32_LIF_NORESET_CYCLES_PER_NEURON)


def _m33_affine(layer: AsyncLayer, act: StepActivity, steps: int) -> np.ndarray:
    if layer.folded:        # one execution per inference on the time-summed input (general kernel), spread over the steps
        e = min(float(layer.cols), float(np.mean(act.active)) * steps)
        return np.array([cpu_sparse_affine_cycles(layer.rows, layer.cols, e, 1.0, weights_in_sram=layer.weights_in_sram) / steps])
    return np.asarray(cpu_sparse_affine_cycles(layer.rows, layer.cols, act.active, act.nonempty(),
                                               weights_in_sram=layer.weights_in_sram), dtype=float)


def _mvp_parts(layer: AsyncLayer, act: StepActivity):
    """(scan/stage on the M33, blocking MVP chain, output loop) of a range-sparse MVP layer, cycles per timestep."""
    flash = not layer.weights_in_sram
    chain = (mvp_launch_cycles(layer.rows) * (EFR32_MVP_FLASH_LAUNCH_FACTOR if flash else 1.0) * act.launches
             + layer.rows * (EFR32_MVP_FLASH_PAIR_FACTOR if flash else 1.0) * act.pairs)
    if layer.hidden:
        total = mvp_sparse_hidden_affine_cycles(layer.rows, layer.cols, act.launches, act.pairs, act.active, flash_weights=flash)
        out = EFR32_MVP_HIDDEN_OUT_CYCLES_PER_ROW * layer.rows
    else:
        total = mvp_sparse_affine_cycles(layer.rows, layer.cols, act.launches, act.pairs, fp32_source=True, flash_weights=flash)
        out = EFR32_MVP_INPUT_OUT_CYCLES_PER_ROW * layer.rows
    return np.asarray(total - chain - out, dtype=float), np.asarray(chain, dtype=float), out


def overlap_cost(layers: List[AsyncLayer], activity: Mapping[int, StepActivity], mvp_layer: Optional[int], *,
                 steps: int) -> AsyncCost:
    """Blocking and async cost (mean cycles per timestep) of the mapping "layer `mvp_layer` range-sparse on the MVP,
    every other layer event-driven on the M33". `mvp_layer=None`: the all-M33 mapping (no async variant)."""
    n = len(layers)
    aff = {i: _m33_affine(layers[i], activity[i], steps) for i in range(n) if i != mvp_layer}
    lif = [_lif(l) for l in layers]
    memsum = 0.0 if layers[-1].folded else EFR32_MEMSUM_CYCLES_PER_NEURON * layers[-1].rows
    if mvp_layer is None:
        tot = sum(float(np.mean(a)) for a in aff.values()) + sum(lif) + memsum
        return AsyncCost(None, tot, None, {"m33_affine": sum(float(np.mean(a)) for a in aff.values()), "lif": sum(lif), "memsum": memsum})
    k, L = mvp_layer, layers[mvp_layer]
    act = activity[k]
    if act.launches is None or act.pairs is None:
        raise ValueError(f"layer {k}: no launches/pairs for the range-sparse MVP candidate")
    scan, chain, out = _mvp_parts(L, act)
    spike_write = EFR32_MVP_FP16_SPIKE_WRITE_CYCLES * L.cols if L.hidden else 0.0
    fold = sum(float(np.mean(aff[i])) for i in aff if layers[i].folded)
    up = sum(aff[i] for i in aff if i < k) + sum(lif[:k]) + spike_write
    down_aff = sum((aff[i] for i in aff if i > k and not layers[i].folded), np.zeros(1))
    blocking = float(np.mean(up + scan + chain + out + down_aff)) + sum(lif[k:]) + memsum + fold
    # --- async: pre + max(chain, tail)
    pre = up + scan + EFR32_ASYNC_START_CYCLES_PER_LAUNCH * act.launches
    chain_a = ((EFR32_ASYNC_CHAIN_FACTOR_SRAM if L.weights_in_sram else EFR32_ASYNC_CHAIN_FACTOR_FLASH) * chain
               + EFR32_ASYNC_GAP_CYCLES_PER_LAUNCH * act.launches)
    tail = (EFR32_ASYNC_TAIL_OUT_FACTOR * out + EFR32_ASYNC_TAIL_LIF_FACTOR * sum(lif[k:])
            + EFR32_ASYNC_TAIL_M33_AFFINE_FACTOR * down_aff + EFR32_ASYNC_TAIL_MEMSUM_FACTOR * memsum)
    step = pre + np.maximum(chain_a, tail)
    wait = np.maximum(chain_a - tail, 0.0)
    notes = []
    if max(np.size(act.launches), *(np.size(a) for a in aff.values()), 1) == 1:
        notes.append("mean activity only: max(E[chain], E[tail]) -- optimistic when chain and tail are close")
    return AsyncCost(k, blocking, float(np.mean(step)) + fold,
                     {"pre": float(np.mean(pre)), "chain_blocking": float(np.mean(chain)), "chain_async": float(np.mean(chain_a)),
                      "tail_blocking": float(np.mean(out + down_aff)) + sum(lif[k:]) + memsum, "tail_async": float(np.mean(tail)),
                      "wait": float(np.mean(wait)), "tail_critical_fraction": float(np.mean(tail >= chain_a)), "fold": fold}, notes)


def layers_of(module, *, m33_sram: Mapping[int, bool] = None, mvp_flash: bool = False, mvp_layer: Optional[int] = None,
              fold_layer: Optional[int] = None) -> List[AsyncLayer]:
    """AsyncLayer list of an Execution IR module's loop-resident [affine, LIF] chain."""
    from spikeduet.backend.schedule_common import layer_index_of
    from spikeduet.ir.execution import AffineOp, LIFUpdateOp
    body = module.loop[0].body
    if any(isinstance(op, AffineOp) for op in module.preheader_ops):
        raise ValueError("async placement: a hoisted affine layer is not supported")
    lifs = {layer_index_of(op.current): op for op in body if isinstance(op, LIFUpdateOp)}
    out = []
    for op in (o for o in body if isinstance(o, AffineOp)):
        i = layer_index_of(op.output)
        if lifs[i].semantics.inhibition:
            raise ValueError(f"async placement: layer {i} uses inhibition")
        sram = (not mvp_flash) if i == mvp_layer else (True if m33_sram is None else bool(m33_sram.get(i, True)))
        out.append(AsyncLayer(op.weight_layer.out_features, op.weight_layer.in_features,
                              lif_reset=lifs[i].semantics.reset_mechanism != "none", weights_in_sram=sram,
                              hidden=i > 0, folded=i == fold_layer))
    return out


def plan_async_placement(module, activity: Mapping[int, StepActivity], *, m33_sram: Mapping[int, bool] = None,
                         mvp_flash_layers=frozenset(), fold_layer: Optional[int] = None) -> dict:
    """Compare the all-M33 mapping with every async-legal single-MVP-layer mapping (layers with launches/pairs in
    `activity`), blocking and async. Returns the per-mapping costs and the cheapest mapping under each execution
    model -- the two can differ, which is what the per-layer planner cannot see."""
    steps = max(1, module.num_timesteps)
    cands = {}
    base = overlap_cost(layers_of(module, m33_sram=m33_sram, fold_layer=fold_layer), activity, None, steps=steps)
    cands["all_m33"] = base
    for k, act in sorted(activity.items()):
        if act.launches is None or k == fold_layer:
            continue
        ls = layers_of(module, m33_sram=m33_sram, mvp_flash=k in mvp_flash_layers, mvp_layer=k, fold_layer=fold_layer)
        cands[f"mvp_layer_{k}"] = overlap_cost(ls, activity, k, steps=steps)
    blocking_best = min(cands, key=lambda c: cands[c].blocking_cycles)
    async_best = min(cands, key=lambda c: cands[c].async_cycles if cands[c].async_cycles is not None else cands[c].blocking_cycles)
    return {"steps": steps, "candidates": cands, "blocking_choice": blocking_best, "async_choice": async_best,
            "choice_changes_under_async": blocking_best != async_best}
