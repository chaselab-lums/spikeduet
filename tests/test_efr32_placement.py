"""Automatic per-layer MVP / Cortex-M33 placement (``engine="auto"``).

The workload tests use synthetic weights in the layer shapes of the shipped calibration profiles
(``tests/fixtures/profiles``): placement depends only on shapes, neuron semantics and activity. They
check that the automatic plan resolves to the same options, and the same generated source, as the
equivalent hand-configured build."""
import dataclasses
import json
import os
from functools import lru_cache

import pytest
import torch

from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.efr32.cost_model import mvp_dense_hidden_affine_cycles
from spikeduet.backend.efr32.placement import (
    M33, MVP_DENSE, MVP_RANGE, ActivityProfile, LayerActivity, PlacementError, estimate_sram, format_placement_report,
    plan_engine_placement)
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, optimized_pipeline
from spikeduet.target.platform import EFR32MG24_MVP
from tests.test_reference import ChainSNN

PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "profiles")

# Layer shapes of the three calibration workloads. Placement depends only on shapes, neuron semantics and
# the activity profile, so synthetic weights reproduce the planner decisions exactly.
_WORKLOADS = {
    "shd": ([(700, 64), (64, 20)], 100),
    "ssc": ([(700, 64), (64, 64), (64, 35)], 100),
    "braille": ([(24, 192), (192, 192), (192, 27)], 51),
}


@lru_cache(maxsize=None)
def _module(workload):
    dims, steps = _WORKLOADS[workload]
    n = len(dims)
    torch.manual_seed(0)
    model = ChainSNN(dims=dims, betas=[0.95] * n, thresholds=[1.0] * n, reset_delays=[False] * n,
                     reset_mechanisms=["subtract"] * (n - 1) + ["none"])
    return compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=steps,
                                          input_temporal_mode=TemporalMode.VARYING, pipeline=optimized_pipeline())


def _profile(workload) -> ActivityProfile:
    return ActivityProfile.load(os.path.join(PROFILES, f"{workload}.json"))


def _gen(module, **kw):
    return Efr32Backend().generate(module, **kw).files


def _engines(plan):
    return {d.layer: d.engine for d in plan.layers}


# ------------------------------------------------------------------------------------------------ cost functions
def test_reconciled_hidden_dense_mvp_cost_matches_measured_layers():
    # measured dense MVP hidden layers with static I/O (cycles per step)
    measured = {(64, 20): 1323, (64, 64): 3421, (64, 35): 2047, (192, 192): 1111240 / 51, (192, 27): 173145 / 51}
    for (n, m), cyc in measured.items():
        assert abs(mvp_dense_hidden_affine_cycles(m, n) - cyc) / cyc < 0.03






def test_estimate_sram_matches_frozen_heap_accounting():
    shd, ssc, br = _module("shd"), _module("ssc"), _module("braille")
    heap = lambda mod, **kw: estimate_sram(mod, static_io_layers=(), m33_weights_in_sram=True, **kw)["heap_requested"]
    assert heap(shd, engine="mvp", m33_layers=(), range_layers=()) == 98000          # dense MVP
    assert heap(shd, engine="mvp", m33_layers=(), range_layers=(0,)) == 99656        # range-sparse input layer
    assert heap(ssc, engine="mvp", m33_layers=(), range_layers=()) == 109662         # dense MVP
    assert heap(ssc, engine="mvp", m33_layers=(), range_layers=(0,)) == 111318       # range-sparse input layer
    assert heap(br, engine="cpu", m33_layers=(0, 1, 2), range_layers=()) == 194112   # all event-driven M33
    assert heap(br, engine="mvp", m33_layers=(), range_layers=()) == 101622          # dense MVP


# ------------------------------------------------------------------------------------------------ reproductions
@pytest.mark.parametrize("workload,expected,manual", [
    ("shd", {0: MVP_RANGE, 1: M33}, dict(engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1}))),
    ("ssc", {0: MVP_RANGE, 1: M33, 2: M33}, dict(engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}))),
    ("braille", {0: M33, 1: M33, 2: M33}, dict(engine="cpu", sparse_cpu=True)),
])
def test_auto_placement_reproduces_measured_builds_byte_identically(workload, expected, manual):
    mod, prof = _module(workload), _profile(workload)
    plan = plan_engine_placement(mod, prof)
    assert _engines(plan) == expected
    assert plan.m33_weights == "sram" and plan.required_defines == ("ISCAS_CPU_SPARSE_WT_SRAM=1",)
    auto = _gen(mod, engine="auto", activity_profile=prof, stage_profile=True)
    hand = _gen(mod, stage_profile=True, **manual)
    assert set(auto) == set(hand) | {"placement_report.json"}
    for name, text in hand.items():
        assert auto[name] == text, name


# ------------------------------------------------------------------------------------------------ fallbacks / overrides
def test_no_profile_keeps_todays_mvp_only_policy():
    mod = _module("shd")
    plan = plan_engine_placement(mod, None)
    assert _engines(plan) == {0: MVP_RANGE, 1: MVP_DENSE} and not plan.m33_layers and not plan.static_io_layers
    assert any("no ActivityProfile" in n for n in plan.notes)
    auto = _gen(mod, engine="auto")
    hand = _gen(mod, engine="mvp", sparse_mvp=True)
    assert auto["snn_generated_mvp.c"] == hand["snn_generated_mvp.c"] and auto["model_data.h"] == hand["model_data.h"]


def test_manual_overrides_pin_layers():
    mod, prof = _module("shd"), _profile("shd")
    pinned_mvp = plan_engine_placement(mod, prof, force_mvp=frozenset({1}))
    d1 = pinned_mvp.layers[1]
    assert (d1.engine, d1.forced) == (MVP_DENSE, "mvp") and pinned_mvp.static_io_layers == {1}
    auto = _gen(mod, engine="auto", activity_profile=prof, force_mvp_layers=frozenset({1}))
    hand = _gen(mod, engine="mvp", sparse_mvp=True, mvp_static_io_layers=frozenset({1}))
    assert auto["snn_generated_mvp.c"] == hand["snn_generated_mvp.c"]
    pinned_m33 = plan_engine_placement(mod, prof, force_m33=frozenset({0}))
    assert _engines(pinned_m33) == {0: M33, 1: M33} and pinned_m33.layers[0].forced == "m33" and pinned_m33.engine == "cpu"
    with pytest.raises(ValueError):
        plan_engine_placement(mod, prof, force_m33=frozenset({1}), force_mvp=frozenset({1}))
    with pytest.raises(ValueError):
        _gen(mod, engine="auto", activity_profile=prof, sparse_cpu=True)      # auto decides that itself
    with pytest.raises(ValueError):
        _gen(mod, engine="mvp", activity_profile=prof)                        # profile only means something in auto
    # explicit engines keep their exact old meaning
    assert _gen(mod, engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1}))["snn_generated_mvp.c"] == \
        _gen(mod, engine="auto", activity_profile=prof)["snn_generated_mvp.c"]


def test_non_binary_activations_are_not_m33_candidates():
    mod, prof = _module("braille"), _profile("braille")
    layers = dict(prof.layers)
    layers[0] = dataclasses.replace(layers[0], binary=False)
    plan = plan_engine_placement(mod, dataclasses.replace(prof, layers=layers))
    assert plan.layers[0].engine != M33 and "non-binary" in plan.layers[0].ineligible[M33]
    assert plan.engine == "mvp" and plan.m33_layers == {1, 2}
    # graded spikes: the IR says layer 1's input is non-binary even if the profile claims otherwise
    model = ChainSNN(dims=[(8, 6), (6, 4)], betas=[0.9, 0.9], thresholds=[1.0, 1.0], graded_spikes_factors=[2.0, 1.0],
                     reset_delays=[False, False])
    gmod = compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=5,
                                          input_temporal_mode=TemporalMode.VARYING, pipeline=optimized_pipeline())
    sparse = LayerActivity(active_inputs=0.1, nonempty_fraction=0.1, binary=True, launches=0.1, pairs=0.1)
    gplan = plan_engine_placement(gmod, ActivityProfile({0: sparse, 1: sparse}, "synthetic", 1, 5))
    assert gplan.layers[1].engine != M33 and "graded" in gplan.layers[1].ineligible[M33]
    assert gplan.layers[0].engine == M33


def test_sram_overflow_falls_back():
    mod, prof = _module("braille"), _profile("braille")
    small = dataclasses.replace(EFR32MG24_MVP, sram_bytes=200 * 1024)            # the all-SRAM plan needs ~201 KB
    auto_plan = plan_engine_placement(mod, prof, platform=small)
    assert auto_plan.sram["total"] <= auto_plan.sram["budget"]
    # per-layer residency: only the cheapest layer to move goes to flash; no global define
    assert auto_plan.m33_weights == "mixed" and auto_plan.required_defines == ()
    assert auto_plan.m33_weight_placement == {0: "sram", 1: "sram", 2: "flash"}
    assert auto_plan.generate_kwargs()["cpu_sparse_weight_placement"] == {0: "sram", 1: "sram", 2: "flash"}
    assert any("flash" in n for n in auto_plan.notes)
    code = _gen(mod, engine="auto", activity_profile=prof, placement_platform=small)["snn_generated_cpu.c"]
    assert "g_csp_wT[2] = iscas_csp_wT2;" in code and "memcpy(wt, iscas_csp_wT1" in code
    all_flash = plan_engine_placement(mod, prof, platform=small, m33_weights="flash")
    assert all_flash.m33_weights == "flash" and all_flash.required_defines == ()
    assert "cpu_sparse_weight_placement" not in all_flash.generate_kwargs()
    assert sum(d.predicted_cycles for d in auto_plan.layers) < sum(d.predicted_cycles for d in all_flash.layers)
    sram_only = plan_engine_placement(mod, prof, platform=small, m33_weights="sram")
    assert sram_only.sram["total"] <= sram_only.sram["budget"]
    assert sram_only.m33_layers < {0, 1, 2} and any("demoted" in n for n in sram_only.notes)
    with pytest.raises(PlacementError):
        plan_engine_placement(mod, prof, platform=dataclasses.replace(EFR32MG24_MVP, sram_bytes=20 * 1024))


def test_hoisted_layer_is_never_a_candidate():
    model = ChainSNN(dims=[(16, 8), (8, 6), (6, 4)], betas=[0.9] * 3, thresholds=[1.0] * 3, reset_delays=[False] * 3)
    mod = compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=10,
                                         input_temporal_mode=TemporalMode.INVARIANT, pipeline=optimized_pipeline())
    sparse = LayerActivity(active_inputs=0.2, nonempty_fraction=0.2, binary=True, launches=0.2, pairs=0.2)
    plan = plan_engine_placement(mod, ActivityProfile({0: sparse, 1: sparse, 2: sparse}, "synthetic", 1, 10))
    assert plan.layers[0].engine == "hoisted" and plan.layers[0].predicted_cycles is None
    assert plan.m33_layers == {1, 2} and plan.engine == "mvp"          # a hoisted affine keeps the MVP engine
    with pytest.raises(ValueError):
        plan_engine_placement(mod, None, force_m33=frozenset({0}))


def test_all_m33_plan_uses_the_cpu_engine():
    model = ChainSNN(dims=[(16, 8), (8, 4)], betas=[0.9] * 2, thresholds=[1.0] * 2, reset_delays=[False] * 2)
    mod = compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=10,
                                         input_temporal_mode=TemporalMode.VARYING, pipeline=optimized_pipeline())
    sparse = LayerActivity(active_inputs=0.3, nonempty_fraction=0.2, binary=True, launches=0.3, pairs=0.3)
    prof = ActivityProfile({0: sparse, 1: sparse}, "synthetic", 1, 10)
    plan = plan_engine_placement(mod, prof)
    assert plan.engine == "cpu" and plan.m33_layers == {0, 1} and plan.generate_kwargs() == {"engine": "cpu", "sparse_cpu": True}
    assert _gen(mod, engine="auto", activity_profile=prof)["snn_generated_cpu.c"] == _gen(mod, engine="cpu", sparse_cpu=True)["snn_generated_cpu.c"]


def test_placement_report_lists_costs_and_margins():
    mod, prof = _module("shd"), _profile("shd")
    report = json.loads(_gen(mod, engine="auto", activity_profile=prof)["placement_report.json"])
    assert report["resolved_generate_kwargs"] == {"engine": "mvp", "sparse_mvp": True, "sparse_mvp_layers": [0], "cpu_sparse_layers": [1]}
    assert report["required_build_defines"] == ["ISCAS_CPU_SPARSE_WT_SRAM=1"] and report["profile_source"] == prof.source
    for layer in report["layers"]:
        assert layer["predicted_cycles_per_step"] > 0 and layer["runner_up"] and layer["margin_pct"] > 0
        assert layer["candidates"][layer["engine"]] == layer["predicted_cycles_per_step"]
    l1 = report["layers"][1]
    assert l1["engine"] == M33 and l1["runner_up"]["engine"] == MVP_DENSE and l1["near_tie"] is True
    assert "NEAR TIE" in format_placement_report(plan_engine_placement(mod, prof))
