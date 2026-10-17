"""Energy oracle (spikeduet/backend/efr32/energy.py): model properties, no hardware."""
import pytest

from spikeduet.backend.efr32.async_placement import AsyncLayer, StepActivity, overlap_cost
from spikeduet.backend.efr32.energy import PG28_PRO_KIT, BoardPower, energy_plan, estimate, mapping_estimates
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, optimized_pipeline
from tests.test_reference import ChainSNN

LAYERS = [AsyncLayer(64, 700, hidden=False), AsyncLayer(64, 64), AsyncLayer(20, 64)]
STEPS = 100


def _act(launches, pairs, e0=60.0, e1=8.0):
    return {0: StepActivity.of_means(e0, 1.0, launches, pairs), 1: StepActivity.of_means(e1), 2: StepActivity.of_means(6.0)}


def test_estimate_is_the_three_state_sum():
    p = PG28_PRO_KIT
    e = estimate(0.040, mvp_busy_s=0.012, wait_s=0.005, power=p)
    assert e.energy_mj == pytest.approx(p.p_cpu_mw * 0.035 + p.p_wait_mw * 0.005 + p.dp_mvp_mw * 0.012)
    assert e.power_mw == pytest.approx(e.energy_mj / 0.040)
    with pytest.raises(ValueError):
        estimate(0.010, wait_s=0.020)


def test_m33_only_mapping_pays_no_mvp_or_wait_term():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), None, steps=STEPS)
    e = mapping_estimates(c, STEPS)
    assert set(e) == {"blocking"}
    assert e["blocking"].mvp_s == 0.0 and e["blocking"].wait_s == 0.0
    assert e["blocking"].energy_mj == pytest.approx(PG28_PRO_KIT.p_cpu_mw * e["blocking"].time_s)
    assert e["blocking"].time_s == pytest.approx(c.blocking_cycles * STEPS / PG28_PRO_KIT.clock_hz)


def test_blocking_cpu_waits_for_the_whole_chain_and_async_keeps_the_mvp_work():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), 0, steps=STEPS)
    e = mapping_estimates(c, STEPS)
    chain = c.parts["chain_blocking"] * STEPS / PG28_PRO_KIT.clock_hz
    assert e["blocking"].wait_s == pytest.approx(chain) and e["blocking"].mvp_s == pytest.approx(chain)
    assert e["async"].mvp_s == pytest.approx(chain)                       # same MVP program time under overlap
    assert e["async"].wait_s == 0.0                                       # fitted convention: async wait is CPU time
    assert e["async"].time_s < e["blocking"].time_s                       # overlap shortens the schedule
    # async saves energy relative to blocking through the shorter time and by no longer polling for the chain
    p = PG28_PRO_KIT
    expected = (p.p_cpu_mw * (e["async"].time_s - e["blocking"].time_s)
                - (p.p_cpu_mw - p.p_wait_mw) * (e["async"].wait_s - e["blocking"].wait_s))
    assert e["async"].energy_mj - e["blocking"].energy_mj == pytest.approx(expected)


def test_async_energy_gain_is_smaller_than_the_latency_gain():
    """The MVP's busy time is paid either way, so a latency saving of x% saves less than x% of the energy."""
    c = overlap_cost(LAYERS, _act(4.0, 60.0), 0, steps=STEPS)
    e = mapping_estimates(c, STEPS)
    dt = 1.0 - e["async"].time_s / e["blocking"].time_s
    de = 1.0 - e["async"].energy_mj / e["blocking"].energy_mj
    assert 0.0 < de < dt


def test_common_power_scale_leaves_relative_deltas_unchanged_but_not_absolute_ones():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), 0, steps=STEPS)
    m33 = overlap_cost(LAYERS, _act(4.0, 60.0), None, steps=STEPS)
    big = PG28_PRO_KIT.scaled(1.28)
    r = lambda p: mapping_estimates(c, STEPS, p)["async"].energy_mj / mapping_estimates(m33, STEPS, p)["blocking"].energy_mj
    assert r(big) == pytest.approx(r(PG28_PRO_KIT))
    d = lambda p: mapping_estimates(c, STEPS, p)["async"].energy_mj - mapping_estimates(m33, STEPS, p)["blocking"].energy_mj
    assert d(big) == pytest.approx(1.28 * d(PG28_PRO_KIT))
    # a different MVP/CPU ratio does change the relative delta: the cross-board claim needs that ratio
    other = BoardPower("what-if", 39e6, 3.0, 5.0, 4.0, 6.0)
    assert r(other) != pytest.approx(r(PG28_PRO_KIT), rel=1e-3)


def test_energy_plan_on_a_module_picks_by_energy_and_reports_deltas():
    model = ChainSNN(dims=[(40, 32), (32, 16), (16, 4)], betas=[0.9] * 3, thresholds=[1.0] * 3)
    mod = compile_source_to_execution_ir(parse_snntorch_model(model), num_timesteps=20, input_temporal_mode=TemporalMode.VARYING,
                                         pipeline=optimized_pipeline())
    act = {0: StepActivity.of_means(5.0, 1.0, 2.0, 4.0), 1: StepActivity.of_means(3.0), 2: StepActivity.of_means(2.0)}
    r = energy_plan(mod, act)
    assert set(r["mappings"]) == {"all_m33", "mvp_layer_0"}
    assert r["baseline"] == "all_m33/blocking"
    assert r["delta_vs_m33"]["all_m33/blocking"] == {"time_pct": 0.0, "energy_pct": 0.0, "energy_mj": 0.0}
    for k in ("mvp_layer_0/blocking", "mvp_layer_0/async"):
        assert k in r["delta_vs_m33"]
    assert r["best_energy"] in r["delta_vs_m33"] and r["best_time"] in r["delta_vs_m33"]


def test_sleeping_while_waiting_saves_only_the_wait_and_never_touches_the_m33_baseline():
    c = overlap_cost(LAYERS, _act(4.0, 60.0), 0, steps=STEPS)
    m33 = overlap_cost(LAYERS, _act(4.0, 60.0), None, steps=STEPS)
    p = PG28_PRO_KIT
    base, sleep = mapping_estimates(c, STEPS, p), mapping_estimates(c, STEPS, p, sleep_while_waiting=True)
    assert sleep["blocking"].time_s == base["blocking"].time_s and sleep["async"].time_s == base["async"].time_s
    chain = c.parts["chain_blocking"] * STEPS / p.clock_hz
    assert base["blocking"].energy_mj - sleep["blocking"].energy_mj == pytest.approx(p.sleep_saving_mw * chain)
    wait = c.parts["wait"] * STEPS / p.clock_hz
    assert base["async"].energy_mj - sleep["async"].energy_mj == pytest.approx(p.sleep_saving_mw * min(wait, base["async"].time_s))
    # the blocking schedule waits for the whole chain, so it gains at least as much as async
    assert (base["blocking"].energy_mj - sleep["blocking"].energy_mj) >= (base["async"].energy_mj - sleep["async"].energy_mj)
    assert mapping_estimates(m33, STEPS, p, sleep_while_waiting=True)["blocking"] == mapping_estimates(m33, STEPS, p)["blocking"]
