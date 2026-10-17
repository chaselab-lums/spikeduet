"""Automatic per-layer engine placement for the EFR32 backend (``engine="auto"``).

``plan_engine_placement`` only resolves the explicit backend options (``sparse_mvp``,
``cpu_sparse_layers``, ``mvp_static_io_layers``, or ``engine="cpu", sparse_cpu=True`` when every
layer lands on the Cortex-M33); code generation then goes through the same paths a hand-configured
build uses, so an automatic plan and the equivalent manual options produce identical source.

Placement depends on runtime activity, so it needs a calibration ``ActivityProfile``
(``profile_activity`` on a training or validation split, never the test split). Without a profile
the plan is ``engine="mvp", sparse_mvp=True``.

Candidates per loop-resident affine, priced in cycles per step:

* ``mvp_range_sparse``: ``mvp_sparse_affine_cycles``, for layers that pass the static range-sparse
  gate (``mvp_sparse_plan``);
* ``mvp_dense``: ``mvp_dense_affine_cycles`` for the input layer, ``mvp_dense_hidden_affine_cycles``
  for hidden layers (realized with static I/O);
* ``m33_event``: ``cpu_sparse_affine_cycles``, only for binary activity (the profile saw only 0/1
  inputs and, for hidden layers, the upstream LIF has ``graded_spikes_factor == 1``).

The lowest predicted cost wins (ties stay on the MVP). The report lists every candidate and the
margin over the runner-up. Hoisted affines run once and are never candidates. ``force_m33`` and
``force_mvp`` pin layers.

SRAM: the plan's heap is computed from the generator's own allocation sites, plus a conservative
static estimate, and checked against ``platform.sram_bytes - sram_reserve_bytes``. With
``m33_weights="auto"`` every Cortex-M33 layer's transposed weights may live in SRAM or in flash
(+2.40 cycles per weight word read); the planner evaluates every assignment, plus demoting M33
layers to their best MVP option, and keeps the cheapest plan that fits, else raises
``PlacementError``. ``"sram"`` / ``"flash"`` force one residency for all M33 layers.

Hidden-neuron reordering (``plan_with_reordering``, or ``allow_reorder=True`` with a profile that
carries ``reorder`` candidates): the order of a hidden layer's neurons is free, and a co-firing
order lowers the range-sparse MVP cost. Each hidden spike edge with a ``ReorderCandidate`` gets one
more candidate, ``mvp_range_sparse_reordered`` (``mvp_sparse_hidden_affine_cycles`` with the
candidate's held-out launches and pairs, plus the upstream LIF's FP16 spike write), with FP16
weights in SRAM or in flash. A plan that selects it must be generated from the permuted model;
``plan_with_reordering`` permutes and re-lowers and returns the module to generate from.
``fold_output=True`` folds a legal output affine, which then runs once per inference on the M33.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from spikeduet.backend.efr32.cost_model import (
    EFR32_MVP_FP16_SPIKE_WRITE_CYCLES,
    mvp_dense_affine_cycles,
    mvp_dense_hidden_affine_cycles,
    mvp_sparse_affine_cycles,
    mvp_sparse_hidden_affine_cycles,
    sparse_merge_gap_pairs,
)
from spikeduet.backend.efr32.cpu_sparse import cpu_sparse_affine_cycles
from spikeduet.backend.efr32.fold import output_fold_plan
from spikeduet.backend.efr32.mvp_emitters import mvp_direct_fused_layer_indices, mvp_sparse_plan
from spikeduet.backend.efr32.mvp_sparse import EFR32_MVP_LAUNCH_CYCLES, EFR32_MVP_MAX_DIM, EFR32_MVP_STAGE_CYCLES_PER_PAIR
from spikeduet.backend.schedule_common import layer_index_of
from spikeduet.ir.execution import AffineOp, ConvertOp, ExecutionModule, LIFUpdateOp
from spikeduet.ir.execution_analysis import producers
from spikeduet.target.platform import EFR32MG24_MVP, PlatformSpec

MVP_RANGE, MVP_DENSE, M33 = "mvp_range_sparse", "mvp_dense", "m33_event"
MVP_RANGE_REORDER = "mvp_range_sparse_reordered"
NEAR_TIE_PCT = 10.0                     # report flag only -- never changes a decision
EFR32_STATIC_SRAM_BASE_BYTES = 6_400    # .stack + .bss + .data without the app's 4*N_in input buffer (upper bound;
                                        # measured 5,800-5,932 B on three reference builds)
EFR32_ALLOC_OVERHEAD_BYTES = 8          # newlib-nano per-allocation overhead estimate


class PlacementError(ValueError):
    """No engine assignment fits the target's SRAM budget."""


# ------------------------------------------------------------------------------------------------ activity profile
@dataclass(frozen=True)
class LayerActivity:
    """Measured activity at the INPUT of one affine layer, per timestep (means over samples x timesteps)."""
    active_inputs: float                 # non-zero inputs per timestep
    nonempty_fraction: float             # fraction of timesteps with >= 1 non-zero input
    binary: bool                         # every non-zero input observed was exactly 1.0
    launches: Optional[float] = None     # merged MVP ranges per timestep (FP32-source merge rule + runtime guard)
    pairs: Optional[float] = None        # processed FP16 column pairs per timestep (same rule)


@dataclass(frozen=True)
class ReorderCandidate:
    """A co-firing order for the neurons feeding one hidden affine layer (affine index i >= 1: the order permutes
    hidden layer i-1), with the range-sparse statistics it gives on a HELD-OUT calibration split."""
    order: Tuple[int, ...]               # new position p holds old neuron order[p]
    launches: float                      # merged MVP ranges per timestep with this order (FP16-source rule + guard)
    pairs: float                         # processed FP16 column pairs per timestep with this order
    launches_as_trained: float           # the same two numbers in the trained neuron order (report only)
    pairs_as_trained: float
    source: str                          # which data fitted the order and which scored it


@dataclass(frozen=True)
class ActivityProfile:
    """Calibration activity for every affine layer (keyed by affine index, as in `affine{i}`)."""
    layers: Mapping[int, LayerActivity]
    source: str                          # provenance: model, dataset split, sample selection
    num_samples: int
    num_timesteps: int
    reorder: Mapping[int, ReorderCandidate] = field(default_factory=dict)   # optional, keyed by affine index

    def to_dict(self) -> dict:
        d = {"source": self.source, "num_samples": self.num_samples, "num_timesteps": self.num_timesteps,
             "layers": {str(k): vars(v) for k, v in sorted(self.layers.items())}}
        if self.reorder:
            d["reorder"] = {str(k): {**vars(v), "order": list(v.order)} for k, v in sorted(self.reorder.items())}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ActivityProfile":
        return cls(layers={int(k): LayerActivity(**v) for k, v in d["layers"].items()}, source=d["source"],
                   num_samples=int(d["num_samples"]), num_timesteps=int(d["num_timesteps"]),
                   reorder={int(k): ReorderCandidate(**{**v, "order": tuple(v["order"])}) for k, v in d.get("reorder", {}).items()})

    @classmethod
    def load(cls, path) -> "ActivityProfile":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


def range_stats(active_pairs: np.ndarray, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per timestep (row of `active_pairs`, a [K, N/2] bool array): merged MVP launches and processed pairs under the
    generated firmware's rule for an FP32-staged source (mvp_emitters.mvp_range_affine_emitter): the
    next active pair joins the current range iff at most `sparse_merge_gap_pairs(rows)` inactive pairs separate them,
    and the runtime guard replaces the ranges by one full-width launch when they would cost more."""
    k, npr = active_pairs.shape
    gap = sparse_merge_gap_pairs(rows)
    act = active_pairs.sum(1).astype(np.int64)
    idx = np.flatnonzero(active_pairs.ravel())
    row, col = idx // npr, idx % npr
    same = row[1:] == row[:-1]
    d = col[1:] - col[:-1] - 1
    new_range = same & (d > gap)
    merged = same & (d > 0) & (d <= gap)
    launches = (act > 0).astype(np.int64) + np.bincount(row[1:][new_range], minlength=k)
    pairs = act + np.bincount(row[1:][merged], weights=d[merged], minlength=k).astype(np.int64)
    per_pair = rows + EFR32_MVP_STAGE_CYCLES_PER_PAIR
    guard = (launches > 0) & (launches * EFR32_MVP_LAUNCH_CYCLES + pairs * per_pair >= EFR32_MVP_LAUNCH_CYCLES + npr * per_pair)
    return np.where(guard, 1, launches), np.where(guard, npr, pairs)


def hidden_range_stats(spikes: np.ndarray, rows: int) -> Tuple[float, float]:
    """Mean launches and processed pairs per timestep of the range-sparse MVP on a hidden spike edge (`spikes`
    [K, N] bool, FP16 source in place): the emitter's merge rule with the FP16 gap, and its runtime guard."""
    npr = spikes.shape[1] // 2
    gap = sparse_merge_gap_pairs(rows, fp16_source=True)
    a = spikes[:, 0::2] | spikes[:, 1::2]
    k = a.shape[0]
    act = a.sum(1).astype(np.int64)
    idx = np.flatnonzero(a.ravel())
    row, col = idx // npr, idx % npr
    same = row[1:] == row[:-1]
    d = col[1:] - col[:-1] - 1
    merged = same & (d > 0) & (d <= gap)
    launches = (act > 0).astype(np.int64) + np.bincount(row[1:][same & (d > gap)], minlength=k)
    pairs = act + np.bincount(row[1:][merged], weights=d[merged], minlength=k).astype(np.int64)
    guard = (launches > 0) & (launches * EFR32_MVP_LAUNCH_CYCLES + pairs * rows >= EFR32_MVP_LAUNCH_CYCLES + npr * rows)
    return float(np.where(guard, 1, launches).mean()), float(np.where(guard, npr, pairs).mean())


def reorder_candidate(fit_spikes: np.ndarray, score_spikes: np.ndarray, rows: int, *, source: str) -> ReorderCandidate:
    """Co-firing order from `fit_spikes`, scored on the disjoint `score_spikes` (both [K, N] bool, one row per
    timestep of the spikes entering the affine layer with `rows` outputs)."""
    from spikeduet.backend.efr32.reorder import cofiring_order
    order = cofiring_order(np.asarray(fit_spikes, dtype=bool))
    score = np.asarray(score_spikes, dtype=bool)
    l1, p1 = hidden_range_stats(score[:, order], rows)
    l0, p0 = hidden_range_stats(score, rows)
    return ReorderCandidate(order=tuple(order), launches=l1, pairs=p1, launches_as_trained=l0, pairs_as_trained=p0, source=source)


@torch.no_grad()
def profile_reordering(source_ir, fit_inputs, score_inputs, *, source: str, max_fit_steps: int = 200_000,
                       batch_size: int = 256) -> Dict[int, ReorderCandidate]:
    """ReorderCandidates for every hidden spike edge with an even input width: the order is fitted on `fit_inputs`
    and scored on the disjoint `score_inputs` (calibration data only -- never the test split)."""
    from spikeduet.ir.model import LIFLayer, LinearLayer
    from spikeduet.reference.simulator import run_sequence
    linears = [layer for layer in source_ir.layers if isinstance(layer, LinearLayer)]
    lifs = [layer for layer in source_ir.layers if isinstance(layer, LIFLayer)]

    def spikes(inputs):
        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            full = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
            inputs = (full[b:b + batch_size] for b in range(0, full.shape[0], batch_size))
        out: Dict[int, List[np.ndarray]] = {}
        for xb in inputs:
            xb = torch.as_tensor(np.asarray(xb), dtype=torch.float32)
            _, _, traces = run_sequence(source_ir, [xb[:, t] for t in range(xb.shape[1])])
            for i in range(1, len(linears)):
                s = torch.stack([tr.spikes[i - 1] for tr in traces]).reshape(-1, linears[i].in_features)
                out.setdefault(i, []).append(s.numpy() != 0)
        return {i: np.concatenate(v) for i, v in out.items()}

    fit, score = spikes(fit_inputs), spikes(score_inputs)
    cands = {}
    for i in range(1, len(linears)):
        n = linears[i].in_features
        if n % 2 or lifs[i - 1].inhibition:
            continue
        f = fit[i]
        if len(f) > max_fit_steps:
            f = f[np.random.default_rng(0).choice(len(f), max_fit_steps, replace=False)]
        cands[i] = reorder_candidate(f, score[i], linears[i].out_features, source=source)
    return cands


def edge_counts(x: np.ndarray, rows: int) -> Dict[str, np.ndarray]:
    """Per-timestep counts for one affine input `x` ([K, N] float, one row per timestep): active inputs, and (N even)
    merged launches / processed pairs. Also whether every non-zero value is exactly 1.0."""
    nz = x != 0
    out = {"active": nz.sum(1), "binary": bool(np.all(x[nz] == 1.0))}
    if x.shape[1] % 2 == 0:
        out["launches"], out["pairs"] = range_stats(nz.reshape(x.shape[0], -1, 2).any(-1), rows)
    return out


@torch.no_grad()
def profile_activity(source_ir, inputs, *, source: str, batch_size: int = 256) -> ActivityProfile:
    """Runs the reference simulator on calibration `inputs`
    ([samples, timesteps, input_dim], or an iterable of such batches) and records the activity entering every
    affine layer: layer 0 sees the external input, layer i > 0 the spikes of LIF i-1. `source` documents the
    calibration data (use a training or validation split, never the test split)."""
    from spikeduet.ir.model import LinearLayer
    from spikeduet.reference.simulator import run_sequence

    linears = [layer for layer in source_ir.layers if isinstance(layer, LinearLayer)]
    if isinstance(inputs, (np.ndarray, torch.Tensor)):
        full = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
        batches: Iterable = (full[b:b + batch_size] for b in range(0, full.shape[0], batch_size))
    else:
        batches = inputs
    acc = {i: {"active": 0.0, "nonempty": 0, "launches": 0.0, "pairs": 0.0, "binary": True, "steps": 0} for i in range(len(linears))}
    s = t_steps = 0
    for xb in batches:
        xb = torch.as_tensor(np.asarray(xb), dtype=torch.float32)
        s, t_steps = s + xb.shape[0], xb.shape[1]
        _, _, traces = run_sequence(source_ir, [xb[:, t] for t in range(t_steps)])
        for i, lin in enumerate(linears):
            edge = xb.transpose(0, 1) if i == 0 else torch.stack([tr.spikes[i - 1] for tr in traces])   # [T, B, N]
            c = edge_counts(edge.reshape(-1, edge.shape[-1]).numpy(), lin.out_features)
            a = acc[i]
            a["active"] += float(c["active"].sum()); a["nonempty"] += int((c["active"] > 0).sum())
            a["binary"] &= c["binary"]; a["steps"] += c["active"].size
            if "launches" in c:
                a["launches"] += float(c["launches"].sum()); a["pairs"] += float(c["pairs"].sum())
    layers = {}
    for i, lin in enumerate(linears):
        a, n = acc[i], acc[i]["steps"]
        even = lin.in_features % 2 == 0
        layers[i] = LayerActivity(active_inputs=a["active"] / n, nonempty_fraction=a["nonempty"] / n, binary=a["binary"],
                                  launches=a["launches"] / n if even else None, pairs=a["pairs"] / n if even else None)
    return ActivityProfile(layers=layers, source=source, num_samples=s, num_timesteps=t_steps)


# ------------------------------------------------------------------------------------------------ plan
@dataclass
class LayerDecision:
    layer: int
    rows: int
    cols: int
    input: str                                       # "external input" | "LIF spikes" | "hoisted (preheader)"
    engine: str                                      # MVP_RANGE | MVP_DENSE | M33 | "hoisted"
    candidates: Dict[str, float] = field(default_factory=dict)       # predicted cycles per timestep
    ineligible: Dict[str, str] = field(default_factory=dict)         # candidate -> reason
    forced: Optional[str] = None                     # "m33" | "mvp" when pinned by a manual override
    notes: List[str] = field(default_factory=list)

    @property
    def predicted_cycles(self) -> Optional[float]:
        return self.candidates.get(self.engine)

    @property
    def runner_up(self) -> Optional[Tuple[str, float]]:
        rest = sorted((v, k) for k, v in self.candidates.items() if k != self.engine)
        return (rest[0][1], rest[0][0]) if rest else None

    @property
    def margin_pct(self) -> Optional[float]:
        """How much more the runner-up would cost, in % of the chosen cost (None if nothing to compare)."""
        ru, cost = self.runner_up, self.predicted_cycles
        return None if ru is None or cost is None else 100.0 * (ru[1] - cost) / cost

    def to_dict(self) -> dict:
        ru, m = self.runner_up, self.margin_pct
        return {"layer": self.layer, "shape": f"{self.cols}->{self.rows}", "input": self.input, "engine": self.engine,
                "predicted_cycles_per_step": None if self.predicted_cycles is None else round(self.predicted_cycles, 1),
                "runner_up": None if ru is None else {"engine": ru[0], "cycles_per_step": round(ru[1], 1)},
                "margin_pct": None if m is None else round(m, 2),
                "near_tie": m is not None and m < NEAR_TIE_PCT,
                "candidates": {k: round(v, 1) for k, v in self.candidates.items()},
                "ineligible": dict(self.ineligible), "forced": self.forced, "notes": list(self.notes)}


@dataclass
class PlacementPlan:
    engine: str                                      # "mvp" | "cpu"
    layers: List[LayerDecision]
    m33_weights: Optional[str]                       # "sram" | "flash" | "mixed" | None (no M33 layer)
    static_io_layers: FrozenSet[int]
    sram: Dict[str, int]
    profile_source: Optional[str]
    notes: List[str] = field(default_factory=list)
    m33_weight_placement: Dict[int, str] = field(default_factory=dict)   # per M33 layer: "sram" | "flash"
    reorder: Dict[int, Tuple[int, ...]] = field(default_factory=dict)    # affine layer -> order of its input neurons
    mvp_flash_weight_layers: FrozenSet[int] = frozenset()                # reordered range-sparse layers reading FP16 W from flash
    fold_output: Optional[int] = None                                    # folded output affine layer, if any

    @property
    def m33_layers(self) -> FrozenSet[int]:
        return frozenset(d.layer for d in self.layers if d.engine == M33)

    @property
    def range_sparse_layers(self) -> FrozenSet[int]:
        return frozenset(d.layer for d in self.layers if d.engine in (MVP_RANGE, MVP_RANGE_REORDER))

    @property
    def reordered_layers(self) -> FrozenSet[int]:
        return frozenset(d.layer for d in self.layers if d.engine == MVP_RANGE_REORDER)

    @property
    def required_defines(self) -> Tuple[str, ...]:
        return ("ISCAS_CPU_SPARSE_WT_SRAM=1",) if self.m33_weights == "sram" else ()

    def generate_kwargs(self) -> dict:
        """The explicit Efr32Backend.generate() options this plan resolves to."""
        mixed = {"cpu_sparse_weight_placement": dict(self.m33_weight_placement)} if self.m33_weights == "mixed" else {}
        if self.fold_output is not None:
            mixed["fold_output_affine"] = True
        if self.engine == "cpu":
            return {"engine": "cpu", "sparse_cpu": True, **mixed}
        kw = {"engine": "mvp", "sparse_mvp": bool(self.range_sparse_layers), **mixed}
        if self.range_sparse_layers:
            kw["sparse_mvp_layers"] = self.range_sparse_layers
        if self.reordered_layers:
            kw["force_sparse_mvp_layers"] = self.reordered_layers
        if self.mvp_flash_weight_layers:
            kw["mvp_flash_weight_layers"] = self.mvp_flash_weight_layers
        if self.m33_layers:
            kw["cpu_sparse_layers"] = self.m33_layers
        if self.static_io_layers:
            kw["mvp_static_io_layers"] = self.static_io_layers
        return kw

    def to_dict(self) -> dict:
        kw = {k: (sorted(v) if isinstance(v, frozenset) else {str(a): b for a, b in v.items()} if isinstance(v, dict) else v)
              for k, v in self.generate_kwargs().items()}
        chosen = [d.predicted_cycles for d in self.layers if d.engine != "hoisted"]
        return {"what": "SpikeDuet automatic engine placement for EFR32; cycles per timestep "
                        "at 39 MHz, per affine layer", "profile_source": self.profile_source,
                "resolved_generate_kwargs": kw, "required_build_defines": list(self.required_defines),
                "m33_weights": self.m33_weights,
                "m33_weight_placement": {str(k): v for k, v in sorted(self.m33_weight_placement.items())},
                **({"reordered_input_neurons_of_layers": sorted(self.reorder)} if self.reorder else {}),
                **({"mvp_flash_weight_layers": sorted(self.mvp_flash_weight_layers)} if self.mvp_flash_weight_layers else {}),
                **({"folded_output_layer": self.fold_output} if self.fold_output is not None else {}),
                "predicted_affine_cycles_per_step": (round(sum(chosen), 1) if chosen and None not in chosen else None),
                "sram_estimate_bytes": dict(self.sram), "layers": [d.to_dict() for d in self.layers],
                "notes": list(self.notes)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)


def format_placement_report(plan: PlacementPlan) -> str:
    """One line per affine layer: chosen engine, predicted cost, runner-up and margin."""
    budget = "unchecked" if plan.sram["budget"] is None else f"{plan.sram['budget']:,}"
    out = [f"engine={plan.engine}  m33_weights={plan.m33_weights}  SRAM est {plan.sram['total']:,} / "
           f"{budget} B  profile: {plan.profile_source}"]
    for d in plan.layers:
        cost = "-" if d.predicted_cycles is None else f"{d.predicted_cycles:,.0f}"
        ru, m = d.runner_up, d.margin_pct
        tail = "" if ru is None or m is None else f" | runner-up {ru[0]} {ru[1]:,.0f} (+{m:.1f}%{', NEAR TIE' if m < NEAR_TIE_PCT else ''})"
        out.append(f"  L{d.layer} {d.cols}->{d.rows:<4} {d.input:18s} {d.engine:17s} {cost:>8}{tail}"
                   + (f" [forced {d.forced}]" if d.forced else ""))
    return "\n".join(out + [f"  note: {n}" for n in plan.notes])


def _upstream_lif(module: ExecutionModule, op: AffineOp) -> Optional[LIFUpdateOp]:
    """The LIFUpdateOp whose spikes feed `op` (directly, or through one ConvertOp), else None (external input)."""
    ops = {o.id: o for o in module.walk_ops()}
    prod = producers(module)
    src = ops.get(prod.get(op.input))
    if isinstance(src, ConvertOp):
        src = ops.get(prod.get(src.input))
    return src if isinstance(src, LIFUpdateOp) else None


def estimate_sram(module: ExecutionModule, *, engine: str, m33_layers: Iterable[int], range_layers: Iterable[int],
                  static_io_layers: Iterable[int], m33_weights_in_sram: bool,
                  m33_sram_layers: Optional[Iterable[int]] = None, mvp_flash_layers: Iterable[int] = ()) -> Dict[str, int]:
    """Heap requested by snn_init(), from the generator's own allocation sites, plus a per-allocation overhead
    estimate and a conservative static estimate
    (base + the app's FP32 input step buffer + static-I/O vectors). `m33_sram_layers` (per-layer residency) overrides
    `m33_weights_in_sram` (one residency for every M33 layer) when given."""
    dims = {layer_index_of(op.output): (op.weight_layer.in_features, op.weight_layer.out_features)
            for op in module.walk_ops() if isinstance(op, AffineOp)}
    inhibited = {layer_index_of(op.current) for op in module.walk_ops()
                 if isinstance(op, LIFUpdateOp) and op.semantics.inhibition}
    m33, rng, sio, mfl = set(m33_layers), set(range_layers), set(static_io_layers), set(mvp_flash_layers)
    heap = allocs = 0
    for l, (n, m) in dims.items():
        if engine == "mvp" and l not in m33:
            heap += (0 if l in mfl else m * n * 2) + n * 2 + m * 2           # g_weight_f16 (unless read from flash),
            allocs += 2 if l in mfl else 3                                   # g_in_f16, g_cur_f16
        heap += m * 4 + n * 4 + m * 4 + m * 4; allocs += 4                   # g_mem, g_stage, g_current, g_spike
        if l in inhibited:
            heap += 2 * m * 4; allocs += 2                                   # g_inhib_mem_state, g_inhib_reset_old
    heap += dims[max(dims)][1] * 4; allocs += 1                              # g_mem_sum
    for l in rng:
        n, m = dims[l]
        heap += 2 * m * 2 + 2 * (n // 2) * 2; allocs += 3                    # lane accumulator, range start/end lists
    for l in m33:
        n, m = dims[l]
        heap += n * 2; allocs += 1                                           # active-index list
        if (l in set(m33_sram_layers)) if m33_sram_layers is not None else m33_weights_in_sram:
            heap += n * m * 4; allocs += 1                                   # SRAM copy of the transposed FP32 W
    input_dim = module.get_buffer(module.input_buffer).shape[-1]
    static = EFR32_STATIC_SRAM_BASE_BYTES + 4 * input_dim + sum(2 * (dims[l][0] + dims[l][1]) for l in sio)
    overhead = EFR32_ALLOC_OVERHEAD_BYTES * allocs
    return {"heap_requested": heap, "allocations": allocs, "alloc_overhead_est": overhead, "static_est": static,
            "total": heap + overhead + static}


def plan_engine_placement(
    module: ExecutionModule,
    profile: Optional[ActivityProfile],
    *,
    force_m33: FrozenSet[int] = frozenset(),
    force_mvp: FrozenSet[int] = frozenset(),
    static_io_override: Optional[FrozenSet[int]] = None,
    platform: PlatformSpec = EFR32MG24_MVP,
    m33_weights: str = "auto",
    sram_reserve_bytes: int = 16 * 1024,
    allow_reorder: bool = False,
    fold_output: bool = False,
    allow_mvp_flash: bool = False,
) -> PlacementPlan:
    """Per-layer engine placement for `module` (see this module's docstring for the rule). `allow_reorder`: also
    consider the reordered range-sparse candidate for layers in `profile.reorder` -- the caller must then generate
    from the permuted model (use `plan_with_reordering`). `fold_output`: fold a legal output affine. `allow_mvp_flash`:
    a range-sparse input layer may also read its FP16 weights from flash (needed when they do not fit SRAM)."""
    force_m33, force_mvp = frozenset(force_m33), frozenset(force_mvp)
    if force_m33 & force_mvp:
        raise ValueError(f"layers {sorted(force_m33 & force_mvp)} are forced onto both the MVP and the M33")
    if m33_weights not in ("auto", "sram", "flash"):
        raise ValueError("m33_weights must be 'auto', 'sram' or 'flash'")
    loop_affines = [op for op in module.loop[0].body if isinstance(op, AffineOp)]
    hoisted = [op for op in module.preheader_ops if isinstance(op, AffineOp)]
    loop_idx = {layer_index_of(op.output) for op in loop_affines}
    bad = (force_m33 | force_mvp) - loop_idx
    if bad:
        raise ValueError(f"forced layers {sorted(bad)} are not loop-resident affine layers (hoisted or nonexistent)")

    static_range = {i for i, p in mvp_sparse_plan(module).items() if not p.fp16_source}
    fused = mvp_direct_fused_layer_indices(module)
    fold = output_fold_plan(module) if fold_output else None      # raises ValueError naming the violated condition
    reorder_cands = dict(profile.reorder) if (allow_reorder and profile is not None) else {}

    def decide(sram_layers: FrozenSet[int], demoted: FrozenSet[int] = frozenset(),
               rr_flash: FrozenSet[int] = frozenset()) -> List[LayerDecision]:
        out = [LayerDecision(layer=layer_index_of(op.output), rows=op.weight_layer.out_features,
                             cols=op.weight_layer.in_features, input="hoisted (preheader)", engine="hoisted",
                             notes=["runs once before the timestep loop; not a placement candidate"]) for op in hoisted]
        for op in loop_affines:
            i, m, n = layer_index_of(op.output), op.weight_layer.out_features, op.weight_layer.in_features
            lif = _upstream_lif(module, op)
            d = LayerDecision(layer=i, rows=m, cols=n, input="LIF spikes" if lif else "external input", engine=MVP_DENSE)
            act = profile.layers.get(i) if profile is not None else None
            # --- MVP candidates
            if lif is None:
                d.candidates[MVP_DENSE] = mvp_dense_affine_cycles(m, n, fp32_source=True, copy=True)
            else:
                d.candidates[MVP_DENSE] = mvp_dense_hidden_affine_cycles(m, n, staged=i not in fused)
            if i in static_range:
                if act is not None and act.launches is not None:
                    d.candidates[MVP_RANGE] = mvp_sparse_affine_cycles(m, n, act.launches, act.pairs, fp32_source=True,
                                                                       flash_weights=i in rr_flash)
                    if i in rr_flash:
                        d.notes.append("range-sparse candidate priced with FP16 weights read from flash")
                else:
                    d.ineligible[MVP_RANGE] = "no calibration launches/pairs for this layer"
            else:
                d.ineligible[MVP_RANGE] = "fails the static range-sparse gate (hidden edge, odd N, or too small)"
            # --- reordered range-sparse MVP: hidden spike edge with a calibrated co-firing order
            if i in reorder_cands:
                if (lif is None or i not in fused or lif.semantics.inhibition or n % 2 or m > EFR32_MVP_MAX_DIM
                        or n // 2 > EFR32_MVP_MAX_DIM or act is None or (fold is not None and i == fold.layer)):
                    d.ineligible[MVP_RANGE_REORDER] = "not a fused, non-inhibited hidden spike edge the range kernel can run"
                else:
                    rc = reorder_cands[i]
                    d.candidates[MVP_RANGE_REORDER] = EFR32_MVP_FP16_SPIKE_WRITE_CYCLES * n + mvp_sparse_hidden_affine_cycles(
                        m, n, rc.launches, rc.pairs, act.active_inputs, flash_weights=i in rr_flash)
                    if i in rr_flash:
                        d.notes.append("reordered range-sparse candidate priced with FP16 weights read from flash")
            # --- M33 candidate
            ir_binary = lif is None or all(g == 1.0 for g in lif.semantics.graded_spikes_factor)
            if act is None:
                d.ineligible[M33] = "no activity profile" if profile is None else "layer missing from the activity profile"
            elif not (act.binary and ir_binary):
                d.ineligible[M33] = ("non-binary activations (graded spikes)" if not ir_binary
                                     else "non-binary activations observed in the calibration profile")
            elif i in demoted:
                d.ineligible[M33] = "demoted to fit SRAM"
            else:
                d.candidates[M33] = cpu_sparse_affine_cycles(m, n, act.active_inputs, act.nonempty_fraction,
                                                             weights_in_sram=i in sram_layers)
                if i not in sram_layers:
                    d.notes.append("M33 weights read in place from flash")
            # --- choice: plain lowest predicted cost, ties stay on the MVP; manual pins win
            mvp_opts = {k: v for k, v in d.candidates.items() if k != M33}
            if fold is not None and i == fold.layer:
                if M33 not in d.candidates:
                    raise PlacementError(f"fold_output: the output affine cannot run on the M33 ({d.ineligible.get(M33)})")
                steps = max(1, module.num_timesteps)
                d.candidates = {M33: cpu_sparse_affine_cycles(m, n, min(float(n), act.active_inputs * steps), 1.0,
                                                              weights_in_sram=i in sram_layers) / steps}
                d.engine, d.forced = M33, "fold"
                d.notes.append("output affine FOLDED: one execution per inference on the M33 (general kernel; cost is an "
                               "estimate spread over the timesteps)")
            elif i in force_m33:
                d.engine, d.forced = M33, "m33"
                if M33 not in d.candidates:
                    d.notes.append(f"forced onto the M33 although {d.ineligible.pop(M33, 'ineligible')}: cost not modelled")
            elif profile is None:
                d.engine = MVP_RANGE if i in static_range else MVP_DENSE      # the static range-sparse policy
                d.candidates.pop(MVP_RANGE, None)
                d.forced = "mvp" if i in force_mvp else None
                d.notes.append("no activity profile: MVP-only static policy")
            elif i in force_mvp:
                d.engine, d.forced = min(mvp_opts, key=mvp_opts.get), "mvp"
            else:
                best_mvp = min(mvp_opts, key=mvp_opts.get)
                d.engine = M33 if M33 in d.candidates and d.candidates[M33] < mvp_opts[best_mvp] else best_mvp
            out.append(d)
        return sorted(out, key=lambda x: x.layer)

    def engine_of(decisions: List[LayerDecision]) -> str:
        return "cpu" if not hoisted and decisions and all(d.engine == M33 for d in decisions) else "mvp"

    def static_io_of(decisions: List[LayerDecision]) -> FrozenSet[int]:
        if static_io_override is not None:
            return frozenset(static_io_override)
        if profile is None:
            return frozenset()
        return frozenset(d.layer for d in decisions if d.engine == MVP_DENSE and d.input == "LIF spikes")

    def sram_of(decisions: List[LayerDecision], sram_layers: FrozenSet[int]) -> Dict[str, int]:
        eng = engine_of(decisions)
        s = estimate_sram(module, engine=eng, m33_layers=[d.layer for d in decisions if d.engine == M33],
                          range_layers=[d.layer for d in decisions if d.engine in (MVP_RANGE, MVP_RANGE_REORDER)],
                          static_io_layers=static_io_of(decisions) if eng == "mvp" else (), m33_weights_in_sram=False,
                          m33_sram_layers=sram_layers,
                          mvp_flash_layers=[d.layer for d in decisions if d.engine in (MVP_RANGE, MVP_RANGE_REORDER) and d.layer in rr_flash_of(decisions)])
        s["budget"] = None if platform.sram_bytes is None else int(platform.sram_bytes) - sram_reserve_bytes
        return s

    def fits(s: Dict[str, int]) -> bool:
        return s["budget"] is None or s["total"] <= s["budget"]

    def total(decisions: List[LayerDecision]) -> float:
        return sum(d.predicted_cycles or 0.0 for d in decisions if d.engine != "hoisted")

    flash_choice: Dict[int, FrozenSet[int]] = {}                     # id(decisions) -> flash set they were priced with
    keep_alive: List[List[LayerDecision]] = []

    def rr_flash_of(decisions: List[LayerDecision]) -> FrozenSet[int]:
        return flash_choice.get(id(decisions), frozenset())

    notes: List[str] = []
    options = []                                                     # feasible (cycles, decisions, sram_layers, sram, note)
    all_layers = frozenset(loop_idx)
    if m33_weights == "sram":
        residencies = [all_layers]
    elif m33_weights == "flash":
        residencies = [frozenset()]
    else:                                                            # per layer: every SRAM/flash assignment
        order = sorted(loop_idx)
        residencies = [frozenset(l for l, keep in zip(order, bits) if keep)
                       for bits in itertools.product((True, False), repeat=len(order))]
    rr_keys = sorted(set(reorder_cands) | (static_range if (allow_mvp_flash and profile is not None) else set()))
    rr_sets = [frozenset(c) for r in range(len(rr_keys) + 1) for c in itertools.combinations(rr_keys, r)]
    for sram_set in residencies:
        for rr_flash in rr_sets:
            decs = decide(sram_set, rr_flash=rr_flash)
            if rr_flash - {d.layer for d in decs if d.engine in (MVP_RANGE, MVP_RANGE_REORDER)}:
                continue                                             # a flash residency for a layer that did not choose range-sparse
            flash_choice[id(decs)] = rr_flash; keep_alive.append(decs)   # ids stay unique while referenced
            s = sram_of(decs, sram_set)
            if fits(s):
                options.append((total(decs), decs, sram_set, s, None))
    if m33_weights != "flash":                                       # repair: all-SRAM with M33 layers demoted to the MVP
        decs = decide(all_layers)
        s = sram_of(decs, all_layers)
        demoted: set = set()
        while not fits(s):
            movable = [d for d in decs if d.engine == M33 and d.forced is None]
            if not movable:
                break
            def penalty(d):
                trial = decide(all_layers, frozenset(demoted | {d.layer}))
                freed = s["total"] - sram_of(trial, all_layers)["total"]
                return (total(trial) - total(decs)) / max(freed, 1)
            demoted.add(min(movable, key=penalty).layer)
            decs = decide(all_layers, frozenset(demoted)); s = sram_of(decs, all_layers)
        if demoted and fits(s):
            options.append((total(decs), decs, all_layers, s, f"SRAM: demoted layers {sorted(demoted)} to the MVP to fit"))
    if not options:
        raise PlacementError(f"no engine placement fits SRAM (budget {int(platform.sram_bytes) - sram_reserve_bytes:,} B)")
    if platform.sram_bytes is None:
        notes.append("platform SRAM unspecified: feasibility NOT checked")
    # cheapest; among equal costs prefer more SRAM-resident layers (a flash choice must buy something)
    cycles, decs, sram_set, s, note = min(options, key=lambda o: (round(o[0], 6), -len(o[2]), len(rr_flash_of(o[1]))))
    chosen_rr = sorted(d.layer for d in decs if d.engine == MVP_RANGE_REORDER)
    if chosen_rr:
        notes.append(f"reordering: input neurons of layers {chosen_rr} reordered (co-firing order from the calibration profile); "
                     "generate from the permuted model (plan_with_reordering)")
    if note:
        notes.append(note)
    m33_of = sorted(d.layer for d in decs if d.engine == M33)
    placement = {l: ("sram" if l in sram_set else "flash") for l in m33_of}
    if any(v == "flash" for v in placement.values()):
        notes.append(f"SRAM: M33 weights of layers {[l for l, v in placement.items() if v == 'flash']} read in place from "
                     "flash (+2.40 cycles per weight word) -- keeping them in SRAM does not fit or is not cheaper")
    if profile is None:
        notes.append("no ActivityProfile: MVP-only plan identical to engine='mvp', sparse_mvp=True (plus any forced M33 layers)")
    if any(d.predicted_cycles is not None and d.margin_pct is not None and d.margin_pct < NEAR_TIE_PCT for d in decs):
        notes.append(f"near tie (< {NEAR_TIE_PCT:.0f}% margin) on layers "
                     f"{[d.layer for d in decs if d.margin_pct is not None and d.margin_pct < NEAR_TIE_PCT]}")
    kinds = set(placement.values())
    residency = None if not placement else kinds.pop() if len(kinds) == 1 else "mixed"
    return PlacementPlan(engine=engine_of(decs), layers=decs, m33_weights=residency,
                         static_io_layers=static_io_of(decs) if engine_of(decs) == "mvp" else frozenset(), sram=s,
                         profile_source=None if profile is None else profile.source, notes=notes,
                         m33_weight_placement=placement,
                         reorder={l: reorder_cands[l].order for l in chosen_rr},
                         mvp_flash_weight_layers=frozenset(d.layer for d in decs if d.engine in (MVP_RANGE, MVP_RANGE_REORDER)
                                                           and d.layer in rr_flash_of(decs)),
                         fold_output=None if fold is None else fold.layer)


def plan_with_reordering(source_ir, profile: ActivityProfile, *, num_timesteps: int, input_temporal_mode=None,
                         pipeline=None, **plan_kwargs):
    """Automatic placement INCLUDING the hidden-neuron reordering rule. Lowers `source_ir`, plans with
    `allow_reorder=True`, and if the plan selected the reordered range-sparse engine for some layer, permutes the
    source model accordingly and lowers it again. Returns (source_ir, module, plan): generate with
    `Efr32Backend().generate(module, **plan.generate_kwargs())`. A plan without reordering returns the inputs'
    own lowering unchanged."""
    from spikeduet.backend.efr32.reorder import permute_hidden_neurons
    from spikeduet.ir.execution import TemporalMode
    from spikeduet.ir.pass_manager import compile_source_to_execution_ir, optimized_pipeline

    def lower(ir):
        return compile_source_to_execution_ir(ir, num_timesteps=num_timesteps,
                                              input_temporal_mode=input_temporal_mode or TemporalMode.VARYING,
                                              pipeline=pipeline or optimized_pipeline())

    module = lower(source_ir)
    plan = plan_engine_placement(module, profile, allow_reorder=True, allow_mvp_flash=True, **plan_kwargs)
    if not plan.reorder:
        return source_ir, module, plan
    for layer, order in sorted(plan.reorder.items()):
        source_ir = permute_hidden_neurons(source_ir, layer - 1, list(order))
    return source_ir, lower(source_ir), plan
