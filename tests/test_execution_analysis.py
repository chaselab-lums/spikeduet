"""Tests for spikeduet.ir.execution_analysis -- def-use queries and
TemporalInvarianceAnalysis.
"""

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.execution_analysis import TemporalInvarianceAnalysis, consumers, producers
from spikeduet.ir.lowering import lower_source_ir
from tests.test_reference import ChainSNN


def _module(dims, num_timesteps=5, input_temporal_mode=TemporalMode.VARYING):
    betas = [0.9] * len(dims)
    thresholds = [1.0] * len(dims)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode)


def test_producers_maps_each_produced_buffer_to_its_op():
    module = _module([(4, 3)])
    p = producers(module)

    assert p["stage0"] == "convert0"
    assert p["current0"] == "affine0"
    assert p["spike0"] == "lif0"
    assert "input" not in p  # model input has no producing op
    assert "mem0" not in p  # membrane is state, not produced


def test_consumers_maps_each_buffer_to_its_readers():
    module = _module([(4, 3)])
    c = consumers(module)

    assert c["input"] == ["convert0"]
    assert c["stage0"] == ["affine0"]
    assert c["current0"] == ["lif0"]
    assert "lif0" in c["mem0"]  # membrane is read by the LIF update that also writes it


def test_temporal_invariance_analysis_mnist_like_propagates_through_layer0():
    module = _module([(784, 64), (64, 32), (32, 10)], input_temporal_mode=TemporalMode.INVARIANT)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert analysis.is_invariant("input")
    assert analysis.is_invariant("stage0")
    assert analysis.is_invariant("current0")
    # membrane/spike are never invariant -- recurrent state
    assert not analysis.is_invariant("mem0")
    assert not analysis.is_invariant("spike0")
    # every layer past 0 depends on a spike buffer -- never invariant
    assert not analysis.is_invariant("stage1")
    assert not analysis.is_invariant("current1")


def test_temporal_invariance_analysis_shd_like_nothing_invariant():
    module = _module([(700, 128), (128, 20)], input_temporal_mode=TemporalMode.VARYING)
    analysis = TemporalInvarianceAnalysis.compute(module)

    assert not analysis.is_invariant("input")
    assert not analysis.is_invariant("stage0")
    assert not analysis.is_invariant("current0")
    assert analysis.invariant_buffers == ()
