"""Energy-per-inference model for EFR32 / EFM32 Series-2 mappings (experimental; no code generation).

Predicts the energy of one inference of an all-M33, blocking-hybrid or async mapping from the latency
model (``async_placement.overlap_cost``) and a table of per-state board powers:

    E = P_cpu * (T - T_wait)  +  P_wait * T_wait  +  dP_mvp * B_mvp

    T       wall time of the inference
    T_wait  time the CPU polls for an MVP program to finish (a tight poll loop draws less than compute)
    B_mvp   time the MVP runs a program; its power adds to the CPU state it overlaps (measured within
            -2.5% .. +6.7% of the sum of the parts)

Fitted on 171 measured inferences of 66 builds on an EFM32PG28 Pro Kit: mean absolute error 2.8%,
leave-one-build-out 2.9% (max 10%).

Two consequences: a faster schedule with the same work saves ``P_cpu * dT`` but runs at a higher
power, so its energy change can be much smaller than its latency change; and an MVP mapping saves
energy only when the work it moves off the CPU is long enough to pay for ``dP_mvp`` over the chain.

The powers belong to one board. Cycle counts match between the PG28 and the EFR32MG24 (same 39 MHz
clock and MVP), so time deltas transfer, but energy deltas transfer only if the MVP/CPU power ratio
is the same. Use a ``BoardPower`` with measured states for another board; ``scaled()`` is a what-if
for a common power scale only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping

from spikeduet.backend.efr32.async_placement import AsyncCost, StepActivity, overlap_cost


@dataclass(frozen=True)
class BoardPower:
    """Steady board power (mW at VMCU) of the execution states an inference is made of."""
    name: str
    clock_hz: float
    p_em1_mw: float        # CPU asleep (WFI), clocks running: the floor a faster schedule does not remove
    p_cpu_mw: float        # CPU running compute kernels (LIF / event accumulate / scan), MVP idle
    p_wait_mw: float       # CPU polling for MVP completion (or a peripheral), MVP idle
    dp_mvp_mw: float       # MVP executing a program: increment over the CPU state it overlaps
    sleep_saving_mw: float = 0.0   # CPU in EM1 instead of polling while an MVP program runs (measured, SL_MVP_POWER_MODE=1)
    note: str = ""

    def scaled(self, factor: float) -> "BoardPower":
        """What-if: every power times `factor` (a different chip at the same cycle counts and the same ratios)."""
        return replace(self, name=f"{self.name} x{factor:g}", p_em1_mw=self.p_em1_mw * factor,
                       p_cpu_mw=self.p_cpu_mw * factor, p_wait_mw=self.p_wait_mw * factor,
                       dp_mvp_mw=self.dp_mvp_mw * factor, sleep_saving_mw=self.sleep_saving_mw * factor)


# Fitted on all measured builds. A power-state microbenchmark measured the same quantities directly:
# CPU LIF 8.0, flash-parameter LIF 7.7, register loop 6.0, polling 6.56, MVP + polling CPU 9.8
# (dp_mvp 3.2-3.35), EM1 3.98 mW.
PG28_PRO_KIT = BoardPower(
    name="PG28 Pro Kit (EFM32PG28B310F1024IM68, 39 MHz HFXO, DC-DC)", clock_hz=39e6, p_em1_mw=3.98,
    p_cpu_mw=7.26, p_wait_mw=6.04, dp_mvp_mw=3.24,
    sleep_saving_mw=1.26,      # MVP + CPU polling 9.52 mW -> MVP + CPU in EM1 8.26 mW; the SDK matvec with
    #                            SL_MVP_POWER_MODE=1: 9.57 -> 8.16 mW at the same throughput
    note="AEM at VMCU 3.29 V; includes board loads (EM1 level is ~3.1 mW above a bare chip)")


@dataclass(frozen=True)
class EnergyEstimate:
    time_s: float
    wait_s: float
    mvp_s: float
    energy_mj: float

    @property
    def power_mw(self) -> float:
        return self.energy_mj / self.time_s if self.time_s > 0 else float("nan")


def estimate(time_s: float, mvp_busy_s: float = 0.0, wait_s: float = 0.0,
             power: BoardPower = PG28_PRO_KIT) -> EnergyEstimate:
    """Energy of one inference from its time decomposition. `wait_s` <= `time_s`; `mvp_busy_s` may exceed `wait_s`
    (async: the MVP runs while the CPU computes)."""
    if wait_s > time_s + 1e-12:
        raise ValueError("wait_s exceeds time_s")
    e = power.p_cpu_mw * (time_s - wait_s) + power.p_wait_mw * wait_s + power.dp_mvp_mw * mvp_busy_s
    return EnergyEstimate(time_s, wait_s, mvp_busy_s, e)


def mapping_estimates(cost: AsyncCost, steps: int, power: BoardPower = PG28_PRO_KIT,
                      sleep_while_waiting: bool = False) -> Dict[str, EnergyEstimate]:
    """Energy of one inference of a mapping priced by `overlap_cost`: keys "blocking" and (for an MVP mapping) "async".
    The all-M33 mapping (cost.mvp_layer is None) has only "blocking" == the plain event-driven M33 schedule.

    `sleep_while_waiting`: the CPU sleeps in EM1 instead of polling while an MVP program runs (`SL_MVP_POWER_MODE=1`
    for the SDK matvec; MVP interrupt + WFI in the raw kernels). Saves `power.sleep_saving_mw` for every second of
    waiting: the whole chain in a blocking schedule, the un-covered part of the chain (`parts["wait"]`) in an async one.
    Measured to cost no latency in the SDK path; the raw-kernel IRQ wake is not implemented."""
    hz = power.clock_hz
    out: Dict[str, EnergyEstimate] = {}
    if cost.mvp_layer is None:
        out["blocking"] = estimate(cost.blocking_cycles * steps / hz, 0.0, 0.0, power)
        return out
    chain = cost.parts["chain_blocking"] * steps / hz              # MVP busy time (blocking chain, no hand-over gaps)
    blk = estimate(cost.blocking_cycles * steps / hz, chain, chain, power)
    if sleep_while_waiting:
        blk = EnergyEstimate(blk.time_s, blk.wait_s, blk.mvp_s, blk.energy_mj - power.sleep_saving_mw * chain)
    out["blocking"] = blk
    if cost.async_cycles is not None:
        t = cost.async_cycles * steps / hz
        # The constants were fitted with the async residual wait counted as CPU time (the poll loops of an async build
        # also run the deferred tail's chunks): charging it at p_wait instead biases async energy low by ~3%.
        asy = estimate(t, chain, 0.0, power)
        if sleep_while_waiting:
            asy = EnergyEstimate(asy.time_s, asy.wait_s, asy.mvp_s,
                                 asy.energy_mj - power.sleep_saving_mw * min(cost.parts["wait"] * steps / hz, t))
        out["async"] = asy
    return out


def energy_plan(module, activity: Mapping[int, StepActivity], *, power: BoardPower = PG28_PRO_KIT, **placement_kw) -> dict:
    """The oracle front door: every async-legal single-MVP-layer mapping of `module` plus the all-M33 mapping, with
    time and energy of the blocking and async schedules and the delta to the all-M33 mapping.

    `activity` / keywords are those of `async_placement.plan_async_placement`. Returns
    {"mappings": {name: {"blocking": EnergyEstimate, "async": EnergyEstimate | None}}, "baseline": "all_m33",
     "delta_vs_m33": {name+"/blocking"|"/async": {"time_pct", "energy_pct", "energy_mj"}}, "best_energy": key, "best_time": key}.
    """
    from spikeduet.backend.efr32.async_placement import plan_async_placement
    plan = plan_async_placement(module, activity, **placement_kw)
    steps = plan["steps"]
    maps, flat = {}, {}
    for name, cost in plan["candidates"].items():
        est = mapping_estimates(cost, steps, power)
        maps[name] = est
        for sched, e in est.items():
            flat[f"{name}/{sched}"] = e
    base = maps["all_m33"]["blocking"]
    delta = {k: {"time_pct": 100.0 * (e.time_s / base.time_s - 1.0), "energy_pct": 100.0 * (e.energy_mj / base.energy_mj - 1.0),
                 "energy_mj": e.energy_mj - base.energy_mj} for k, e in flat.items()}
    return {"steps": steps, "mappings": maps, "baseline": "all_m33/blocking", "delta_vs_m33": delta,
            "best_energy": min(flat, key=lambda k: flat[k].energy_mj), "best_time": min(flat, key=lambda k: flat[k].time_s),
            "power": power.name}


def layer_energy_split(layers, activity: Mapping[int, StepActivity], mvp_layer: int, steps: int,
                       power: BoardPower = PG28_PRO_KIT) -> Dict[str, float]:
    """Diagnostic: where the async-vs-M33 energy difference comes from (mJ per inference), for one MVP layer:
    cpu_time_saved (P_cpu * dT of the schedule), mvp_added (dP_mvp * B_mvp) and the wait term."""
    m33 = overlap_cost(layers, activity, None, steps=steps)
    hyb = overlap_cost(layers, activity, mvp_layer, steps=steps)
    a = mapping_estimates(hyb, steps, power)
    b = mapping_estimates(m33, steps, power)["blocking"]
    asy = a.get("async", a["blocking"])
    return {"dt_ms": 1e3 * (b.time_s - asy.time_s), "cpu_time_saved_mj": power.p_cpu_mw * (b.time_s - asy.time_s),
            "mvp_added_mj": power.dp_mvp_mw * asy.mvp_s, "wait_saved_mj": -(power.p_cpu_mw - power.p_wait_mw) * asy.wait_s,
            "delta_mj": asy.energy_mj - b.energy_mj}
