"""Tests for InvariantAffineHoistingPass, LICM legality,
preheader verifier support, and workload-level structural tests.

No dataset-name, layer-index, architecture-specific, or timestep-count
special case in these tests -- all structural expectations emerge from
the dataflow analysis.
"""

import dataclasses

import pytest

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    LIFUpdateOp,
    TemporalMode,
)
from spikeduet.ir.execution_analysis import TemporalInvarianceAnalysis
from spikeduet.ir.execution_pretty import format_execution_ir
from spikeduet.ir.execution_verifier import (
    ExecutionIRVerificationError,
    verify_execution_ir,
)
from spikeduet.ir.licm import (
    InvariantAffineHoistingPass,
    is_hoistable,
    should_hoist,
)
from spikeduet.ir.lowering import lower_source_ir
from spikeduet.ir.pass_manager import (
    PassManager,
    VerifyExecutionIRPass,
    compile_source_to_execution_ir,
    temporal_licm_pipeline,
)
from tests.test_reference import ChainSNN


def _module(dims, num_timesteps=10, input_temporal_mode=TemporalMode.VARYING):
    betas = [0.9] * len(dims)
    thresholds = [1.0] * len(dims)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode)


def _run_licm(module):
    """Run the LICM pipeline and return (transformed_module, pass_instance)."""
    p = InvariantAffineHoistingPass()
    pipeline = PassManager([VerifyExecutionIRPass(), p, VerifyExecutionIRPass()])
    result = pipeline.run(module)
    return result, p


# ===================================================================
# LICM legality
# ===================================================================


def test_hoistable_affine_with_invariant_input():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    affine = [op for op in module.walk_ops() if isinstance(op, AffineOp)][0]
    assert is_hoistable(affine, analysis)


def test_not_hoistable_affine_with_varying_input():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    analysis = TemporalInvarianceAnalysis.compute(module)

    affine = [op for op in module.walk_ops() if isinstance(op, AffineOp)][0]
    assert not is_hoistable(affine, analysis)


def test_lif_never_hoistable():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    lif = [op for op in module.walk_ops() if isinstance(op, LIFUpdateOp)][0]
    assert not is_hoistable(lif, analysis)


def test_should_hoist_affine_not_convert():
    """should_hoist is True for AffineOp, False for ConvertOp
    (ConvertOp is demand-driven, not independently hoisted)."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    convert = [op for op in module.walk_ops() if isinstance(op, ConvertOp)][0]
    affine = [op for op in module.walk_ops() if isinstance(op, AffineOp)][0]

    assert should_hoist(affine, analysis) is True
    assert should_hoist(convert, analysis) is False  # demand-driven only


# ===================================================================
# Hoisting positive: static input -> Affine hoisted
# ===================================================================


def test_hoisting_positive_static_input():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    assert len(result.preheader_ops) > 0
    preheader_types = [type(op).__name__ for op in result.preheader_ops]
    assert "AffineOp" in preheader_types
    assert p.stats.affines_hoisted == 1


# ===================================================================
# Hoisting dependency chain: static input -> Convert -> Affine
# ===================================================================


def test_hoisting_dependency_chain():
    """Both ConvertOp and AffineOp are hoisted when the Convert feeds
    the hoisted Affine (demand-driven)."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    preheader_types = [type(op).__name__ for op in result.preheader_ops]
    assert preheader_types == ["ConvertOp", "AffineOp"]
    assert p.stats.converts_hoisted == 1
    assert p.stats.affines_hoisted == 1

    # LIF remains in loop
    loop_types = [type(op).__name__ for op in result.loop[0].body]
    assert "LIFUpdateOp" in loop_types
    assert "AffineOp" not in loop_types  # only affine was in layer 0


# ===================================================================
# Hoisting negative: temporal input -> no hoisting
# ===================================================================


def test_hoisting_negative_temporal_input():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    result, p = _run_licm(module)

    assert len(result.preheader_ops) == 0
    assert p.stats.affines_hoisted == 0
    assert p.stats.converts_hoisted == 0


# ===================================================================
# Stateful barrier: LIF spike -> Affine must not hoist
# ===================================================================


def test_stateful_barrier_prevents_hoisting():
    """An affine that consumes a spike buffer (LIF output) is never
    hoisted, even if the model's external input is invariant."""
    module = _module([(4, 3), (3, 2)], input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    # Only the first affine (fed by invariant input) is hoisted
    assert p.stats.affines_hoisted == 1
    assert p.stats.affines_considered == 2

    # Second affine (fed by spike0) remains in loop
    loop_affines = [op for op in result.loop[0].body if isinstance(op, AffineOp)]
    assert len(loop_affines) == 1
    assert loop_affines[0].id == "affine1"


# ===================================================================
# Mixed dependency: any varying operand -> stays temporal
# ===================================================================


def test_mixed_dependency_stays_in_loop():
    """If an op has any varying runtime operand, it must remain in the
    temporal loop."""
    module = _module([(4, 3), (3, 2), (2, 1)],
                     input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    # Only affine0 should be hoisted (its input is invariant)
    assert p.stats.affines_hoisted == 1
    # affine1 and affine2 consume spike buffers -> varying -> stay in loop
    loop_affines = [op for op in result.loop[0].body if isinstance(op, AffineOp)]
    assert len(loop_affines) == 2


# ===================================================================
# Arbitrary depth
# ===================================================================


def test_arbitrary_depth_hoisting():
    """The pass does not assume two layers -- works for any depth."""
    module = _module([(10, 8), (8, 6), (6, 4), (4, 2)],
                     input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    # Only the first affine is hoisted (all later ones depend on spikes)
    assert p.stats.affines_hoisted == 1
    assert p.stats.affines_considered == 4

    # All 4 LIFs remain in loop
    loop_lifs = [op for op in result.loop[0].body if isinstance(op, LIFUpdateOp)]
    assert len(loop_lifs) == 4

    # 3 affines remain in loop
    loop_affines = [op for op in result.loop[0].body if isinstance(op, AffineOp)]
    assert len(loop_affines) == 3


# ===================================================================
# Workload-level structural tests (MNIST-like, SHD-like)
# ===================================================================


def test_mnist_like_structural():
    """MNIST-like: 784->64->32->10, static input, 10 timesteps.
    Expected: first affine hoisted, others not."""
    module = _module([(784, 64), (64, 32), (32, 10)],
                     num_timesteps=10,
                     input_temporal_mode=TemporalMode.INVARIANT)
    result, p = _run_licm(module)

    # First Convert+Affine hoisted
    assert p.stats.affines_hoisted == 1
    assert p.stats.converts_hoisted == 1
    assert p.stats.affines_considered == 3

    # Preheader has convert0 + affine0
    preheader_ids = [op.id for op in result.preheader_ops]
    assert preheader_ids == ["convert0", "affine0"]

    # Loop body: lif0, convert1, affine1, lif1, convert2, affine2, lif2
    loop_ids = [op.id for op in result.loop[0].body]
    assert "lif0" in loop_ids
    assert "affine1" in loop_ids
    assert "affine2" in loop_ids
    assert "convert0" not in loop_ids
    assert "affine0" not in loop_ids

    # Verify the transformed module passes verification
    verify_execution_ir(result)


def test_shd_like_structural():
    """SHD-like: 700->64->20, temporal input, 100 timesteps.
    Expected: 0 affine ops hoisted."""
    module = _module([(700, 64), (64, 20)],
                     num_timesteps=100,
                     input_temporal_mode=TemporalMode.VARYING)
    result, p = _run_licm(module)

    # No hoisting
    assert p.stats.affines_hoisted == 0
    assert p.stats.converts_hoisted == 0
    assert len(result.preheader_ops) == 0

    # All ops remain in loop
    assert len(result.loop[0].body) == len(module.loop[0].body)


# ===================================================================
# Pretty-printer structural tests
# ===================================================================


def test_mnist_pretty_print_after_hoisting():
    module = _module([(784, 64), (64, 10)],
                     num_timesteps=10,
                     input_temporal_mode=TemporalMode.INVARIANT)
    result, _ = _run_licm(module)
    text = format_execution_ir(result)

    assert "Preheader:" in text
    assert "convert %input -> %stage0" in text
    assert "affine  %stage0 -> %current0" in text
    # LIF should be in the TemporalLoop section
    assert "TemporalLoop" in text
    assert "lif" in text


def test_shd_pretty_print_empty_preheader():
    module = _module([(700, 64), (64, 10)],
                     num_timesteps=100,
                     input_temporal_mode=TemporalMode.VARYING)
    result, _ = _run_licm(module)
    text = format_execution_ir(result)

    assert "Preheader:" in text
    assert "<empty>" in text


# ===================================================================
# Pass statistics
# ===================================================================


def test_pass_statistics_deterministic():
    module = _module([(784, 64), (64, 32), (32, 10)],
                     input_temporal_mode=TemporalMode.INVARIANT)

    _, p1 = _run_licm(module)
    _, p2 = _run_licm(module)

    assert p1.stats.affines_hoisted == p2.stats.affines_hoisted
    assert p1.stats.converts_hoisted == p2.stats.converts_hoisted
    assert p1.stats.invariant_buffers == p2.stats.invariant_buffers
    assert p1.stats.varying_buffers == p2.stats.varying_buffers


def test_pass_statistics_counts():
    module = _module([(4, 3), (3, 2)],
                     input_temporal_mode=TemporalMode.INVARIANT)
    _, p = _run_licm(module)

    assert p.stats.affines_considered == 2
    assert p.stats.affines_hoisted == 1
    assert p.stats.converts_hoisted == 1
    assert p.stats.invariant_buffers == 3  # input, stage0, current0
    assert p.stats.varying_buffers > 0


# ===================================================================
# Analysis invalidation
# ===================================================================


def test_analysis_invalidation_after_hoisting():
    """After hoisting, re-running the analysis on the transformed module
    should give consistent results (same invariance facts)."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    result, _ = _run_licm(module)

    # Recompute analysis on transformed module
    analysis_after = TemporalInvarianceAnalysis.compute(result)

    assert analysis_after.is_invariant("input")
    assert analysis_after.is_invariant("stage0")
    assert analysis_after.is_invariant("current0")
    assert not analysis_after.is_invariant("mem0")
    assert not analysis_after.is_invariant("spike0")


# ===================================================================
# Preheader verifier: positive
# ===================================================================


def test_verifier_accepts_valid_preheader():
    """A module with correctly hoisted ops passes verification."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    result, _ = _run_licm(module)
    verify_execution_ir(result)  # must not raise


# ===================================================================
# Preheader verifier: malformed preheader tests
# ===================================================================


def test_verifier_rejects_lif_in_preheader():
    """LIFUpdateOp in preheader is rejected."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    loop = module.loop[0]

    # Manually put the LIF in the preheader (illegal)
    lif_op = loop.body[2]
    assert isinstance(lif_op, LIFUpdateOp)

    broken = dataclasses.replace(
        module,
        preheader_ops=(lif_op,),
        loop=(dataclasses.replace(loop, body=loop.body[:2]),),
    )

    with pytest.raises(ExecutionIRVerificationError, match="not allowed in preheader"):
        verify_execution_ir(broken)


def test_verifier_rejects_varying_input_in_preheader():
    """Preheader op consuming a VARYING external input is rejected."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    loop = module.loop[0]

    # Manually put convert0 (which reads the VARYING input) in preheader
    convert_op = loop.body[0]
    assert isinstance(convert_op, ConvertOp)
    assert convert_op.input == "input"

    broken = dataclasses.replace(
        module,
        preheader_ops=(convert_op,),
        loop=(dataclasses.replace(loop, body=loop.body[1:]),),
    )

    with pytest.raises(ExecutionIRVerificationError, match="varying external input"):
        verify_execution_ir(broken)


def test_verifier_rejects_varying_input_affine_in_preheader():
    """Preheader AffineOp consuming a VARYING external input is rejected.
    (Correction #7 test case)."""
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.VARYING)
    loop = module.loop[0]

    # Manually put affine0 in preheader, rewired to read directly from
    # the VARYING input buffer (skipping convert).
    affine_op = loop.body[1]
    assert isinstance(affine_op, AffineOp)
    bad_affine = dataclasses.replace(affine_op, input="input")

    broken = dataclasses.replace(
        module,
        preheader_ops=(bad_affine,),
        loop=(dataclasses.replace(loop, body=(loop.body[0],) + loop.body[2:]),),
    )

    with pytest.raises(ExecutionIRVerificationError, match="varying external input"):
        verify_execution_ir(broken)


def test_verifier_rejects_preheader_consuming_loop_produced_buffer():
    """A preheader op that tries to consume a buffer produced only by a
    loop-body op is rejected (def-use violation)."""
    module = _module([(4, 3), (3, 2)], input_temporal_mode=TemporalMode.INVARIANT)
    loop = module.loop[0]

    # affine1 reads stage1, which is produced by convert1 in the loop body.
    # Try to put affine1 in the preheader without convert1.
    affine1 = loop.body[4]
    assert affine1.id == "affine1"

    remaining = list(loop.body)
    remaining.pop(4)

    broken = dataclasses.replace(
        module,
        preheader_ops=(affine1,),
        loop=(dataclasses.replace(loop, body=tuple(remaining)),),
    )

    with pytest.raises(ExecutionIRVerificationError, match="before it is defined"):
        verify_execution_ir(broken)


def test_verifier_accepts_empty_preheader():
    """An empty preheader is always valid."""
    module = _module([(4, 3)])
    assert module.preheader_ops == ()
    verify_execution_ir(module)  # must not raise


# ===================================================================
# temporal_licm_pipeline integration
# ===================================================================


def test_temporal_licm_pipeline_integration():
    """The opt-in temporal_licm_pipeline produces a correct transformed
    module."""
    ir = parse_snntorch_model(
        ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    )
    module = compile_source_to_execution_ir(
        ir,
        num_timesteps=5,
        input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=temporal_licm_pipeline(),
    )

    assert len(module.preheader_ops) == 2  # convert0, affine0
    assert len(module.loop[0].body) == 1   # lif0 only


def test_default_pipeline_does_not_hoist():
    """The default pipeline does NOT run LICM -- preheader stays empty."""
    ir = parse_snntorch_model(
        ChainSNN(dims=[(4, 3)], betas=[0.9], thresholds=[1.0])
    )
    module = compile_source_to_execution_ir(
        ir,
        num_timesteps=5,
        input_temporal_mode=TemporalMode.INVARIANT,
    )

    assert module.preheader_ops == ()
    assert len(module.loop[0].body) == 3  # convert0, affine0, lif0


# ===================================================================
# walk_ops includes preheader
# ===================================================================


def test_walk_ops_includes_preheader_ops():
    module = _module([(4, 3)], input_temporal_mode=TemporalMode.INVARIANT)
    result, _ = _run_licm(module)

    all_ops = result.walk_ops()
    # Preheader (convert0, affine0) + loop (lif0)
    assert len(all_ops) == 3
    assert all_ops[0].id == "convert0"
    assert all_ops[1].id == "affine0"
    assert all_ops[2].id == "lif0"


# ===================================================================
# layer_sequence works after hoisting (backend compatibility)
# ===================================================================


def test_layer_sequence_after_hoisting():
    """c_common.layer_sequence still extracts the correct layer
    sequence from a transformed module."""
    from spikeduet.backend.c_common import layer_sequence

    module = _module([(4, 3), (3, 2)], input_temporal_mode=TemporalMode.INVARIANT)
    result, _ = _run_licm(module)

    linear_layers, lif_specs = layer_sequence(result)

    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == 4
    assert linear_layers[0].out_features == 3
    assert linear_layers[1].in_features == 3
    assert linear_layers[1].out_features == 2
    assert len(lif_specs) == 2
