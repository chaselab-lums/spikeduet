"""Overlap-aware placement cost (spikeduet/backend/efr32/async_placement.py): model properties, no hardware."""
import numpy as np
import pytest

from spikeduet.backend.efr32.async_placement import AsyncLayer, StepActivity, layers_of, overlap_cost, plan_async_placement
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, optimized_pipeline
from tests.test_reference import ChainSNN

LAYERS = [AsyncLayer(64, 700, hidden=False), AsyncLayer(64, 64), AsyncLayer(20, 64)]


def _act(launches, pairs, e1=8.0):
    return {0: StepActivity.of_means(60.0, 1.0, launches, pairs), 1: StepActivity.of_means(e1), 2: StepActivity.of_means(6.0)}


def test_all_m33_mapping_has_no_async_variant():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), None, steps=100)
    assert c.async_cycles is None and c.blocking_cycles > 0


def test_async_step_is_pre_plus_max_of_chain_and_tail():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), 0, steps=100)
    p = c.parts
    assert c.async_cycles == pytest.approx(p["pre"] + max(p["chain_async"], p["tail_async"]))
    assert p["wait"] == pytest.approx(max(p["chain_async"] - p["tail_async"], 0.0))
    assert c.async_cycles < c.blocking_cycles                     # chain and tail overlap
    short = overlap_cost(LAYERS, _act(1.0, 4.0), 0, steps=100)    # short chain: the deferred tail is the critical path
    assert short.parts["tail_critical_fraction"] == 1.0 and short.parts["wait"] == 0.0
    assert short.async_cycles > short.blocking_cycles - short.parts["chain_blocking"]   # deferral overhead is charged


def test_per_step_activity_is_never_cheaper_than_its_mean():
    rng = np.random.default_rng(0)
    L = rng.integers(0, 9, 400).astype(float); P = L * 15.0
    steps = {0: StepActivity(np.full(400, 60.0), L, P), 1: StepActivity(np.full(400, 8.0)), 2: StepActivity(np.full(400, 6.0))}
    a = overlap_cost(LAYERS, steps, 0, steps=100)
    b = overlap_cost(LAYERS, _act(float(L.mean()), float(P.mean())), 0, steps=100)
    assert a.async_cycles >= b.async_cycles - 1e-6 and b.notes   # E[max] >= max(E); the mean form says it is optimistic
    assert a.blocking_cycles == pytest.approx(b.blocking_cycles)


def test_plan_on_a_module_and_missing_range_stats():
    model = ChainSNN(dims=[(40, 32), (32, 16), (16, 4)], betas=[0.9] * 3, thresholds=[1.0] * 3)
    mod = compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=20, input_temporal_mode=TemporalMode.VARYING,
                                         pipeline=optimized_pipeline())
    assert [(l.rows, l.cols, l.hidden) for l in layers_of(mod)] == [(32, 40, False), (16, 32, True), (4, 16, True)]
    act = {0: StepActivity.of_means(5.0, 1.0, 2.0, 4.0), 1: StepActivity.of_means(3.0), 2: StepActivity.of_means(2.0)}
    r = plan_async_placement(mod, act)
    assert set(r["candidates"]) == {"all_m33", "mvp_layer_0"} and r["blocking_choice"] in r["candidates"]
    with pytest.raises(ValueError, match="no launches/pairs"):
        overlap_cost(layers_of(mod), act, 1, steps=20)
