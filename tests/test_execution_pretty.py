"""Tests for spikeduet.ir.execution_pretty -- deterministic Execution
IR text dump.
"""

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.execution_pretty import format_execution_ir
from spikeduet.ir.lowering import lower_source_ir
from tests.test_reference import ChainSNN


def _module(dims, input_temporal_mode=TemporalMode.VARYING, num_timesteps=10):
    betas = [0.9] * len(dims)
    thresholds = [1.0] * len(dims)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode)


def test_format_is_deterministic():
    module = _module([(4, 3), (3, 2)])

    text_a = format_execution_ir(module)
    text_b = format_execution_ir(module)

    assert text_a == text_b


def test_format_includes_timesteps_and_buffer_kinds():
    module = _module([(4, 3)])
    text = format_execution_ir(module)

    assert "timesteps=10" in text
    assert "%input" in text
    assert "%stage0" in text
    assert "%current0" in text
    assert "%mem0" in text
    assert "%spike0" in text
    assert "state" in text  # membrane buffer tag


def test_format_distinguishes_invariant_vs_varying_input():
    invariant_module = _module([(784, 64), (64, 10)], input_temporal_mode=TemporalMode.INVARIANT)
    varying_module = _module([(700, 128), (128, 10)], input_temporal_mode=TemporalMode.VARYING)

    invariant_text = format_execution_ir(invariant_module)
    varying_text = format_execution_ir(varying_module)

    input_line_invariant = [l for l in invariant_text.splitlines() if "%input " in l][0]
    input_line_varying = [l for l in varying_text.splitlines() if "%input " in l][0]

    assert "invariant" in input_line_invariant
    assert "varying" in input_line_varying


def test_format_never_dumps_raw_weight_values():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)
    weight_value = ir.layers[0].weight.detach().reshape(-1).tolist()[0]
    module = lower_source_ir(ir, num_timesteps=1)

    text = format_execution_ir(module)

    assert f"{weight_value:.6f}" not in text


def test_format_shows_op_sequence():
    module = _module([(4, 3), (3, 2)])
    text = format_execution_ir(module)

    assert "convert %input -> %stage0" in text
    assert "affine  %stage0 -> %current0" in text
    assert "lif     %current0, %mem0 -> %spike0" in text
    assert "convert %spike0 -> %stage1" in text
