"""Tests for generalized TemporalInvarianceAnalysis.

Tests the dataflow-driven invariance propagation through ConvertOp/
AffineOp chains, the varying barrier at LIFUpdateOp, and the varying_buffers
set.

Does NOT test any dataset-name, layer-index, architecture-specific, or
timestep-count special case -- all invariance results emerge from
dataflow analysis.
"""

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    LIFUpdateOp,
    TemporalMode,
    op_is_pure,
    op_outputs,
    op_reads_state,
    op_runtime_inputs,
    op_writes_state,
)
from spikeduet.ir.execution_analysis import TemporalInvarianceAnalysis
from spikeduet.ir.lowering import lower_source_ir
from tests.test_reference import ChainSNN


def _module(dims, num_timesteps=5, input_temporal_mode=TemporalMode.VARYING):
    betas = [0.9] * len(dims)
    thresholds = [1.0] * len(dims)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode)


# ---------------------------------------------------------------
# Operation purity traits
# ---------------------------------------------------------------


def test_affine_op_is_pure():
    module = _module([(4, 3)])
    ops = module.walk_ops()
    affine = [op for op in ops if isinstance(op, AffineOp)][0]
    assert op_is_pure(affine) is True
    assert op_reads_state(affine) is False
    assert op_writes_state(affine) is False


def test_convert_op_is_pure():
    module = _module([(4, 3)])
    ops = module.walk_ops()
    convert = [op for op in ops if isinstance(op, ConvertOp)][0]
    assert op_is_pure(convert) is True
    assert op_reads_state(convert) is False
    assert op_writes_state(convert) is False


def test_lif_update_op_is_stateful():
    module = _module([(4, 3)])
    ops = module.walk_ops()
    lif = [op for op in ops if isinstance(op, LIFUpdateOp)][0]
    assert op_is_pure(lif) is False
    assert op_reads_state(lif) is True
    assert op_writes_state(lif) is True


def test_op_runtime_inputs_affine():
    module = _module([(4, 3)])
    affine = [op for op in module.walk_ops() if isinstance(op, AffineOp)][0]
    inputs = op_runtime_inputs(affine)
    assert inputs == ("stage0",)  # weight_layer is compile-time, not runtime


def test_op_runtime_inputs_lif():
    module = _module([(4, 3)])
    lif = [op for op in module.walk_ops() if isinstance(op, LIFUpdateOp)][0]
    inputs = op_runtime_inputs(lif)
    assert inputs == ("current0", "mem0")


def test_op_outputs_all_types():
    module = _module([(4, 3)])
    ops = module.walk_ops()
    convert = [op for op in ops if isinstance(op, ConvertOp)][0]
    affine = [op for op in ops if isinstance(op, AffineOp)][0]
    lif = [op for op in ops if isinstance(op, LIFUpdateOp)][0]

    assert op_outputs(convert) == ("stage0",)
    assert op_outputs(affine) == ("current0",)
    assert op_outputs(lif) == ("spike0",)


# ---------------------------------------------------------------
# Invariance seed: static external input -> invariant
# ---------------------------------------------------------------


def test_static_external_input_is_invariant():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert analysis.is_invariant("input")


# ---------------------------------------------------------------
# Invariance seed: temporal external input -> varying
# ---------------------------------------------------------------


def test_temporal_external_input_is_varying():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert not analysis.is_invariant("input")
    assert "input" in analysis.varying_buffers


# ---------------------------------------------------------------
# Propagation: static input -> Convert -> invariant
# ---------------------------------------------------------------


def test_static_input_through_convert_remains_invariant():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert analysis.is_invariant("stage0")  # convert output


# ---------------------------------------------------------------
# Propagation: static input -> Affine -> invariant
# ---------------------------------------------------------------


def test_static_input_through_affine_remains_invariant():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert analysis.is_invariant("current0")  # affine output


# ---------------------------------------------------------------
# Propagation: temporal input -> Affine -> varying
# ---------------------------------------------------------------


def test_temporal_input_through_affine_is_varying():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert not analysis.is_invariant("stage0")
    assert not analysis.is_invariant("current0")


# ---------------------------------------------------------------
# LIF output -> always varying
# ---------------------------------------------------------------


def test_lif_spike_output_always_varying():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert not analysis.is_invariant("spike0")
    assert "spike0" in analysis.varying_buffers


# ---------------------------------------------------------------
# Membrane state -> always varying
# ---------------------------------------------------------------


def test_membrane_state_always_varying():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert not analysis.is_invariant("mem0")
    assert "mem0" in analysis.varying_buffers


# ---------------------------------------------------------------
# Static chain remains invariant until stateful op
# ---------------------------------------------------------------


def test_static_chain_invariant_until_lif():
    """In a multi-layer model with static input, the first layer's
    Convert+Affine chain is invariant; everything after the LIF is varying."""
    module = _module([(4, 3), (3, 2)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    # First layer: invariant chain
    assert analysis.is_invariant("input")
    assert analysis.is_invariant("stage0")
    assert analysis.is_invariant("current0")

    # LIF breaks the chain
    assert not analysis.is_invariant("mem0")
    assert not analysis.is_invariant("spike0")

    # Second layer: everything depends on spike0 -> varying
    assert not analysis.is_invariant("stage1")
    assert not analysis.is_invariant("current1")
    assert not analysis.is_invariant("mem1")
    assert not analysis.is_invariant("spike1")


# ---------------------------------------------------------------
# MNIST-like full analysis
# ---------------------------------------------------------------


def test_mnist_like_full_invariance_analysis():
    module = _module([(784, 64), (64, 32), (32, 10)],
                     num_timesteps=10,
                     input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    # Only first layer's Convert+Affine chain is invariant
    assert analysis.is_invariant("input")
    assert analysis.is_invariant("stage0")
    assert analysis.is_invariant("current0")

    # Everything else is varying
    for name in ["mem0", "spike0", "stage1", "current1", "mem1", "spike1",
                 "stage2", "current2", "mem2", "spike2"]:
        assert not analysis.is_invariant(name), f"{name} should be varying"


# ---------------------------------------------------------------
# SHD-like full analysis
# ---------------------------------------------------------------


def test_shd_like_nothing_invariant():
    module = _module([(700, 128), (128, 20)],
                     num_timesteps=100,
                     input_temporal_mode=TemporalMode.VARYING)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert analysis.invariant_buffers == ()
    assert len(analysis.varying_buffers) > 0


# ---------------------------------------------------------------
# Varying buffers tracking
# ---------------------------------------------------------------


def test_varying_buffers_count():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    # invariant: input, stage0, current0
    assert len(analysis.invariant_buffers) == 3
    # varying: mem0, spike0
    assert len(analysis.varying_buffers) == 2


def test_all_buffers_classified():
    """Every buffer is classified as either invariant or varying."""
    module = _module([(4, 3), (3, 2)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    all_names = {b.name for b in module.buffers}
    classified = set(analysis.invariant_buffers) | set(analysis.varying_buffers)
    assert classified == all_names


# ---------------------------------------------------------------
# Preheader ops in analysis
# ---------------------------------------------------------------


def test_analysis_works_on_module_with_preheader():
    """TemporalInvarianceAnalysis.compute works correctly on a module
    that already has ops in the preheader (post-transformation)."""
    import dataclasses

    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    loop = module.loop[0]

    # Manually hoist convert0 and affine0 to preheader
    preheader = (loop.body[0], loop.body[1])  # convert0, affine0
    remaining = (loop.body[2],)  # lif0
    new_loop = dataclasses.replace(loop, body=remaining)
    transformed = dataclasses.replace(module, preheader_ops=preheader, loop=(new_loop,))

    analysis = TemporalInvarianceAnalysis.compute(transformed)

    assert analysis.is_invariant("input")
    assert analysis.is_invariant("stage0")
    assert analysis.is_invariant("current0")
    assert not analysis.is_invariant("mem0")
    assert not analysis.is_invariant("spike0")


# ---------------------------------------------------------------
# Arbitrary depth
# ---------------------------------------------------------------


def test_arbitrary_depth_invariance():
    """Invariance analysis works for any depth, not just 2 layers."""
    module = _module([(10, 8), (8, 6), (6, 4), (4, 2)],
                     input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    # Only first layer chain is invariant
    assert analysis.is_invariant("input")
    assert analysis.is_invariant("stage0")
    assert analysis.is_invariant("current0")

    # All subsequent layers are varying
    for i in range(1, 4):
        assert not analysis.is_invariant(f"stage{i}")
        assert not analysis.is_invariant(f"current{i}")
