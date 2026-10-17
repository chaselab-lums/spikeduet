"""hil/aem.py: burst location and energy extraction on synthetic AEM traces (no board).

The trace model matches the measured AEM: 6,250 Hz boxcar samples, ~23 uA noise per sample, 3.291 V. A replay burst
is flanked by two 15 ms EM1 gaps; `locate_burst` must find it from the known spacing alone."""
import numpy as np
import pytest

from tools.hil.aem import GAP_MS, locate_burst

DT_US = 160.0
V = 3.291


def _trace(levels, seed=0, noise_uA=23.0):
    """Levels: list of (duration_s, current_mA | callable(n) -> array). Returns t_us, i_mA, v_V."""
    rng = np.random.default_rng(seed)
    parts = []
    for dur, lvl in levels:
        n = int(round(dur * 1e6 / DT_US))
        parts.append(lvl(n) if callable(lvl) else np.full(n, float(lvl)))
    i = np.concatenate(parts) + rng.normal(0.0, noise_uA * 1e-3, sum(len(p) for p in parts))
    t = 233_000_000.0 + np.arange(len(i)) * DT_US
    return t, i, np.full(len(i), V)


def _expected_power(level_mA):
    return level_mA * V


@pytest.mark.parametrize("burst_mA", [2.6, 1.45, 1.25])
def test_burst_power_is_recovered_even_when_close_to_the_gap_level(burst_mA):
    burst_s = 0.60
    gap = GAP_MS * 1e-3
    t, i, v = _trace([(1.2, 2.1), (gap, 1.19), (burst_s, burst_mA), (gap, 1.19), (0.9, 2.1)])
    loc = locate_burst(t, i, v, burst_s)
    assert loc["p_burst_mW"] == pytest.approx(_expected_power(burst_mA), rel=0.01)
    assert loc["spacing_s"] == pytest.approx(burst_s + gap, abs=5 * DT_US * 1e-6)
    assert loc["i_gap_mA"] == pytest.approx(1.19, abs=0.02)
    assert loc["i_poll_mA"] == pytest.approx(2.1, abs=0.05)


def test_alternating_burst_with_low_dips_is_not_mistaken_for_a_gap():
    """A dense-MVP replay burst: the CPU waits in EM1 for the MVP in ~1 ms pieces. No 15 ms run sits at the gap
    level, so the spacing rule still picks the real gaps."""
    burst_s = 0.50
    gap = GAP_MS * 1e-3

    def alt(n):
        x = np.full(n, 2.4)
        x[(np.arange(n) % 12) < 5] = 1.30          # ~0.8 ms dips toward the EM1 level
        return x

    t, i, v = _trace([(1.0, 2.1), (gap, 1.19), (burst_s, alt), (gap, 1.19), (1.0, 2.1)], seed=3)
    loc = locate_burst(t, i, v, burst_s)
    mean_level = (7 * 2.4 + 5 * 1.30) / 12
    assert loc["p_burst_mW"] == pytest.approx(_expected_power(mean_level), rel=0.02)
    assert loc["spacing_s"] == pytest.approx(burst_s + gap, abs=5 * DT_US * 1e-6)


def test_burst_position_is_free_in_the_trace():
    burst_s = 0.30
    gap = GAP_MS * 1e-3
    for lead in (0.8, 1.7, 2.9):
        t, i, v = _trace([(lead, 2.0), (gap, 1.2), (burst_s, 2.8), (gap, 1.2), (0.7, 2.0)], seed=int(lead * 10))
        loc = locate_burst(t, i, v, burst_s)
        assert loc["p_burst_mW"] == pytest.approx(_expected_power(2.8), rel=0.01)
        assert loc["t_burst_start_us"] == pytest.approx(t[0] + (lead + gap) * 1e6, abs=6 * DT_US)


def test_too_short_trace_is_rejected():
    t, i, v = _trace([(0.05, 2.0)])
    with pytest.raises(ValueError):
        locate_burst(t, i, v, 0.60)
