"""Tests for spikeduet.ir.lowering -- Source IR -> Execution IR.

Covers lowering correctness (op order, buffer shapes, membrane/spike
buffers, temporal region), temporal-input semantics (MNIST-like static vs
SHD-like time-varying), and all Leaky semantic preservation through
LIFExecutionSpec.
"""

import pytest

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import (
    AffineOp,
    BufferKind,
    ConvertOp,
    LIFUpdateOp,
    StorageKind,
    TemporalMode,
)
from spikeduet.ir.lowering import lower_source_ir
from tests.test_reference import ChainSNN


def test_single_layer_lowering_op_order_and_shapes():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=5)

    assert module.num_timesteps == 5
    assert module.input_buffer == "input"
    assert module.output_buffer == "spike0"
    assert module.get_buffer("input").shape == (4,)
    assert module.get_buffer("spike0").shape == (3,)

    ops = module.walk_ops()
    assert len(ops) == 3
    assert isinstance(ops[0], ConvertOp) and ops[0].input == "input" and ops[0].output == "stage0"
    assert isinstance(ops[1], AffineOp) and ops[1].input == "stage0" and ops[1].output == "current0"
    assert ops[1].weight_layer.in_features == 4 and ops[1].weight_layer.out_features == 3
    assert isinstance(ops[2], LIFUpdateOp)
    assert ops[2].current == "current0" and ops[2].membrane == "mem0" and ops[2].output_spike == "spike0"


def test_deep_chain_lowering_op_order():
    model = ChainSNN(dims=[(4, 6), (6, 4), (4, 2)], betas=[0.95, 0.9, 0.92], thresholds=[1.0, 0.8, 1.1])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=10)

    ops = module.walk_ops()
    assert len(ops) == 9  # 3 layers x (convert, affine, lif)
    kinds = [type(op).__name__ for op in ops]
    assert kinds == [
        "ConvertOp", "AffineOp", "LIFUpdateOp",
        "ConvertOp", "AffineOp", "LIFUpdateOp",
        "ConvertOp", "AffineOp", "LIFUpdateOp",
    ]

    # layer i's convert reads the previous layer's spike buffer
    assert ops[3].input == "spike0" and ops[3].output == "stage1"
    assert ops[6].input == "spike1" and ops[6].output == "stage2"
    assert module.output_buffer == "spike2"


def test_membrane_buffers_are_stateful_others_are_not():
    model = ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=1)

    for b in module.buffers:
        if b.kind is BufferKind.MEMBRANE:
            assert b.stateful is True
        else:
            assert b.stateful is False


def test_num_timesteps_defaults_to_source_num_steps_if_set():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)
    ir.num_steps = 7

    module = lower_source_ir(ir)

    assert module.num_timesteps == 7


def test_missing_num_timesteps_raises():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    with pytest.raises(ValueError, match="num_timesteps"):
        lower_source_ir(ir)


def test_non_positive_num_timesteps_raises():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    with pytest.raises(ValueError, match="positive"):
        lower_source_ir(ir, num_timesteps=0)


def test_mnist_like_static_input_marked_invariant():
    model = ChainSNN(dims=[(784, 64), (64, 32), (32, 10)], betas=[0.95] * 3, thresholds=[1.0] * 3)
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=10, input_temporal_mode=TemporalMode.INVARIANT)

    assert module.get_buffer("input").temporal is TemporalMode.INVARIANT


def test_shd_like_temporal_input_marked_varying():
    model = ChainSNN(dims=[(700, 128), (128, 20)], betas=[0.95, 0.95], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=10, input_temporal_mode=TemporalMode.VARYING)

    assert module.get_buffer("input").temporal is TemporalMode.VARYING


def test_default_input_temporal_mode_is_conservative_varying():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=1)

    assert module.get_buffer("input").temporal is TemporalMode.VARYING


def test_membrane_and_spike_buffers_never_invariant_even_under_invariant_input():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=1, input_temporal_mode=TemporalMode.INVARIANT)

    assert module.get_buffer("mem0").temporal is TemporalMode.VARYING
    assert module.get_buffer("spike0").temporal is TemporalMode.VARYING


def test_all_buffers_default_dtype_f32():
    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=1)

    for b in module.buffers:
        assert b.dtype is StorageKind.F32


def test_lif_execution_spec_preserves_l0_through_l5_semantics():
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0)
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5]],
        reset_mechanisms=["zero"],
        reset_delays=[True],
        inhibitions=[True],
        state_quants=[q],
    )
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=1)
    lif_op = [op for op in module.walk_ops() if isinstance(op, LIFUpdateOp)][0]
    spec = lif_op.semantics

    assert spec.num_neurons == 4
    assert list(spec.beta) == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert list(spec.threshold) == pytest.approx([0.3, 0.6, 0.9, 1.2])
    assert list(spec.graded_spikes_factor) == pytest.approx([0.25, 0.75, 1.5, 2.5])
    assert spec.reset_mechanism == "zero"
    assert spec.reset_delay is True
    assert spec.inhibition is True
    assert spec.state_quant is not None
    assert len(spec.state_quant.levels) == 4


def test_small_two_layer_topology_lowers_cleanly():
    """45 -> 6 -> 3."""
    model = ChainSNN(dims=[(45, 6), (6, 3)], betas=[0.95, 0.95], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)

    module = lower_source_ir(ir, num_timesteps=10)

    assert module.get_buffer("input").shape == (45,)
    assert module.get_buffer("spike0").shape == (6,)
    assert module.output_buffer == "spike1"
    assert module.get_buffer("spike1").shape == (3,)


def test_lowering_rejects_source_ir_without_verify_module_ir_style_mismatch():
    """lower_source_ir raises a ValueError (not a silent wrong lowering) on
    an unpaired Linear/LIF count; lowering checks this itself because it does
    not call verify_module_ir (see that function's docstring)."""
    from spikeduet.ir.model import SNNModel

    model = ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)
    broken = SNNModel(layers=[ir.layers[0]])  # Linear with no following LIF

    with pytest.raises(ValueError, match="Linear -> Leaky"):
        lower_source_ir(broken, num_timesteps=1)
