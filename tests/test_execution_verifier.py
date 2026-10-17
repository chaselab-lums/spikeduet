"""Tests for spikeduet.ir.execution_verifier -- structural verification
of the Execution IR, exercised via both valid lowered modules and
deliberately corrupted hand-built ones.
"""

import dataclasses

import pytest

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution_verifier import ExecutionIRVerificationError, verify_execution_ir
from spikeduet.ir.lowering import lower_source_ir
from tests.test_reference import ChainSNN


def _valid_module(dims=((4, 3),), num_timesteps=3):
    betas = [0.9] * len(dims)
    thresholds = [1.0] * len(dims)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps)


def test_valid_lowered_module_passes():
    module = _valid_module()
    verify_execution_ir(module)  # must not raise


def test_deep_valid_lowered_module_passes():
    module = _valid_module(dims=((4, 6), (6, 4), (4, 2)))
    verify_execution_ir(module)  # must not raise


def test_missing_buffer_reference_rejected():
    module = _valid_module()
    ops = list(module.loop[0].body)
    bad_affine = dataclasses.replace(ops[1], input="does_not_exist")
    ops[1] = bad_affine
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=tuple(ops)),))

    with pytest.raises(ExecutionIRVerificationError, match="undefined buffer"):
        verify_execution_ir(broken)


def test_affine_shape_mismatch_rejected():
    """Corrupting stage0's shape is caught by the earliest check that sees
    it -- ConvertOp's own size-agreement check (convert0 also reads/writes
    stage0, and runs before affine0 in program order). AffineOp's OWN
    in_features check is exercised in isolation by giving affine0 a
    same-size-as-declared-but-wrong-vs-weight input, below."""
    module = _valid_module()
    bad_input_buf = dataclasses.replace(module.get_buffer("stage0"), shape=(999,))
    new_buffers = tuple(bad_input_buf if b.name == "stage0" else b for b in module.buffers)
    broken = dataclasses.replace(module, buffers=new_buffers)

    with pytest.raises(ExecutionIRVerificationError, match="has size"):
        verify_execution_ir(broken)


def test_affine_in_features_mismatch_isolated_from_convert_rejected():
    """Isolates AffineOp's own in_features check: shrink ONLY the
    ConvertOp's declared output size expectation by instead feeding
    affine0 directly from the (correctly-sized) model input, bypassing
    stage0 -- input has 4 elements but weight_layer.in_features is 4 too,
    so widen the mismatch by using a 2-layer model and pointing layer 1's
    affine at the (differently-sized) input buffer instead of its own
    stage1."""
    module = _valid_module(dims=((4, 3), (3, 2)))
    ops = list(module.loop[0].body)
    # ops layout: convert0, affine0, lif0, convert1, affine1, lif1
    affine1 = ops[4]
    assert affine1.id == "affine1"
    ops[4] = dataclasses.replace(affine1, input="input")  # size 4, but weight_layer.in_features == 3
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=tuple(ops)),))

    with pytest.raises(ExecutionIRVerificationError, match="in_features"):
        verify_execution_ir(broken)


def test_affine_output_shape_mismatch_rejected():
    module = _valid_module()
    bad_output_buf = dataclasses.replace(module.get_buffer("current0"), shape=(999,))
    new_buffers = tuple(bad_output_buf if b.name == "current0" else b for b in module.buffers)
    broken = dataclasses.replace(module, buffers=new_buffers)

    with pytest.raises(ExecutionIRVerificationError, match="out_features"):
        verify_execution_ir(broken)


def test_lif_current_shape_mismatch_rejected():
    module = _valid_module()
    bad_buf = dataclasses.replace(module.get_buffer("current0"), shape=(999,))
    new_buffers = tuple(bad_buf if b.name == "current0" else b for b in module.buffers)
    broken = dataclasses.replace(module, buffers=new_buffers)

    with pytest.raises(ExecutionIRVerificationError, match="current"):
        verify_execution_ir(broken)


def test_lif_membrane_not_stateful_rejected():
    module = _valid_module()
    bad_mem = dataclasses.replace(module.get_buffer("mem0"), stateful=False)
    new_buffers = tuple(bad_mem if b.name == "mem0" else b for b in module.buffers)
    broken = dataclasses.replace(module, buffers=new_buffers)

    with pytest.raises(ExecutionIRVerificationError, match="stateful"):
        verify_execution_ir(broken)


def test_undefined_buffer_consumed_before_definition_rejected():
    module = _valid_module()
    ops = list(module.loop[0].body)
    # swap convert0 and affine0 -- affine0 now consumes stage0 before convert0 defines it
    ops[0], ops[1] = ops[1], ops[0]
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=tuple(ops)),))

    with pytest.raises(ExecutionIRVerificationError, match="before it is defined"):
        verify_execution_ir(broken)


def test_convert_size_mismatch_rejected():
    module = _valid_module()
    bad_buf = dataclasses.replace(module.get_buffer("stage0"), shape=(999,))
    new_buffers = tuple(bad_buf if b.name == "stage0" else b for b in module.buffers)
    broken = dataclasses.replace(module, buffers=new_buffers)

    with pytest.raises(ExecutionIRVerificationError):
        verify_execution_ir(broken)


def test_empty_loop_body_rejected():
    module = _valid_module()
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=()),))

    with pytest.raises(ExecutionIRVerificationError, match="empty body"):
        verify_execution_ir(broken)


def test_non_positive_trip_count_rejected():
    module = _valid_module()
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], trip_count=0),))

    with pytest.raises(ExecutionIRVerificationError, match="trip_count"):
        verify_execution_ir(broken)


def test_duplicate_op_id_rejected():
    module = _valid_module()
    ops = list(module.loop[0].body)
    ops[1] = dataclasses.replace(ops[1], id=ops[0].id)
    broken = dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=tuple(ops)),))

    with pytest.raises(ExecutionIRVerificationError, match="duplicate op id"):
        verify_execution_ir(broken)


def test_wrong_input_buffer_kind_rejected():
    module = _valid_module()
    broken = dataclasses.replace(module, input_buffer="mem0")

    with pytest.raises(ExecutionIRVerificationError, match="INPUT"):
        verify_execution_ir(broken)


def test_wrong_output_buffer_kind_rejected():
    module = _valid_module()
    broken = dataclasses.replace(module, output_buffer="mem0")

    with pytest.raises(ExecutionIRVerificationError, match="SPIKE"):
        verify_execution_ir(broken)
