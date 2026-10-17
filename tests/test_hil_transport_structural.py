"""Structural checks of runtime/efr32/app.c's two protocol branches, from its source text.

They guard against regressions such as moving the ACK printf into a timed region or leaking a
temporal control line into the static branch, without needing a board.
"""
from pathlib import Path

import pytest

APP_C_PATH = Path(__file__).resolve().parents[1] / "runtime" / "efr32" / "app.c"


@pytest.fixture(scope="module")
def app_c_branches():
    """Splits app.c's app_process_action() ifdef into its two branch
    texts, using the same '#if defined(ISCAS_INPUT_TEMPORAL) ... #else
    ... #endif' markers the file itself already has -- one temporal
    region, one static region."""
    text = APP_C_PATH.read_text()
    start = text.index("void app_process_action(void)")
    body = text[start:]

    temporal_marker = "#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL"
    else_marker = "\n#else\n"
    endif_marker = "\n#endif\n"

    t0 = body.index(temporal_marker) + len(temporal_marker)
    e0 = body.index(else_marker, t0)
    temporal_region = body[t0:e0]

    s0 = e0 + len(else_marker)
    e1 = body.index(endif_marker, s0)
    static_region = body[s0:e1]

    return temporal_region, static_region


def test_temporal_path_emits_ready_t0(app_c_branches):
    temporal, _ = app_c_branches
    assert 'printf("READY_T:0\\n")' in temporal


def test_temporal_path_emits_one_ack_per_timestep_with_matching_index(app_c_branches):
    temporal, _ = app_c_branches
    # the ACK printf must be inside the "for (int t = 0; ..." loop and use
    # `t` (the loop's own timestep index) as its format argument -- not a
    # hardcoded/mismatched value.
    for_idx = temporal.index("for (int t = 0; t < ISCAS_STEPS; t++)")
    ack_idx = temporal.index('printf("ACK_T:%d\\n", t)')
    assert ack_idx > for_idx, "ACK_T printf must be inside the per-timestep loop"


def test_temporal_path_emits_bad_header_error(app_c_branches):
    temporal, _ = app_c_branches
    assert "ERR:BAD_HEADER" in temporal


def test_temporal_path_still_emits_uart_timeout_error(app_c_branches):
    temporal, _ = app_c_branches
    assert "ERR:UART_TIMEOUT" in temporal


def test_temporal_path_reads_and_validates_header_before_ready(app_c_branches):
    temporal, _ = app_c_branches
    header_idx = temporal.index("steps_hdr")
    ready_idx = temporal.index('printf("READY_T:0\\n")')
    assert header_idx < ready_idx, "header must be read/validated before READY_T:0 is sent"


def test_static_path_does_not_emit_timestep_acks(app_c_branches):
    _, static = app_c_branches
    assert "ACK_T" not in static
    assert "READY_T" not in static
    assert "BEGIN_TEMPORAL" not in static


def test_static_path_still_emits_uart_timeout_error(app_c_branches):
    _, static = app_c_branches
    assert "ERR:UART_TIMEOUT" in static


def test_static_path_unchanged_pred_format(app_c_branches):
    """Backward-compatibility guard: the static branch must still compute
    prediction via the single-shot run_snn_inference() contract, not the
    streaming iscas_infer_* functions the temporal branch uses."""
    _, static = app_c_branches
    assert "run_snn_inference(input_buffer)" in static
    assert "iscas_infer_step" not in static
    assert "iscas_infer_reset" not in static


def test_ack_printf_is_outside_both_dwt_timing_brackets(app_c_branches):
    """ACK transmission must never be counted as parse_us or infer_us --
    model compute timing stays separate from transport. Structural proxy: the ACK printf must appear AFTER the
    infer_cycles accumulation line for iteration t (i.e. outside the [i0,
    i1] bracket), not between i0 and the accumulation."""
    temporal, _ = app_c_branches
    accum_idx = temporal.index("infer_cycles += (i1 - i0);")
    ack_idx = temporal.index('printf("ACK_T:%d\\n", t)')
    assert ack_idx > accum_idx
