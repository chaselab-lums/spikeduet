"""Tests for spikeduet.ir.verifier -- the standalone source-IR
verifier, exercised both via hand-built IR values (the whole point of a
verifier independent of the frontend) and via Frontend V1 output (which
must always pass).
"""

import math

import pytest
import torch

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel, StateQuantSpec
from spikeduet.ir.verifier import IRVerificationError, verify_module_ir
from tests.test_reference import ChainSNN


def _linear(in_f, out_f, bias=True):
    weight = torch.randn(out_f, in_f)
    b = torch.randn(out_f) if bias else None
    return LinearLayer(name="lin", in_features=in_f, out_features=out_f, weight=weight, bias=b)


def _lif(beta=None, threshold=None, reset_mechanism="subtract", reset_delay=False,
         graded_spikes_factor=None, inhibition=False, state_quant=None):
    return LIFLayer(
        name="lif",
        beta=beta if beta is not None else [0.9],
        threshold=threshold if threshold is not None else [1.0],
        reset_mechanism=reset_mechanism,
        reset_delay=reset_delay,
        graded_spikes_factor=graded_spikes_factor if graded_spikes_factor is not None else [1.0],
        inhibition=inhibition,
        state_quant=state_quant,
    )


def test_valid_hand_built_ir_passes():
    model = SNNModel(layers=[_linear(4, 3), _lif(), _linear(3, 2), _lif()])
    verify_module_ir(model)  # must not raise


def test_frontend_output_always_passes():
    model = ChainSNN(dims=[(4, 6), (6, 3)], betas=[0.95, 0.9], thresholds=[1.0, 0.8])
    ir = parse_snntorch_model(model)
    verify_module_ir(ir)  # parse_snntorch_model already calls this; re-checking is a no-op


def test_empty_model_rejected():
    with pytest.raises(IRVerificationError):
        verify_module_ir(SNNModel(layers=[]))


def test_must_start_with_linear():
    with pytest.raises(IRVerificationError, match="must begin with a Linear"):
        verify_module_ir(SNNModel(layers=[_lif()]))


def test_must_end_with_lif():
    with pytest.raises(IRVerificationError, match="must end with a LIF"):
        verify_module_ir(SNNModel(layers=[_linear(4, 3)]))


def test_two_linears_in_a_row_rejected():
    model = SNNModel(layers=[_linear(4, 3), _linear(3, 2), _lif()])
    with pytest.raises(IRVerificationError, match="expected a LIF"):
        verify_module_ir(model)


def test_two_lifs_in_a_row_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(), _lif()])
    with pytest.raises(IRVerificationError, match="expected a Linear"):
        verify_module_ir(model)


def test_dimension_mismatch_between_linears_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(), _linear(5, 2), _lif()])
    with pytest.raises(IRVerificationError, match="output dimension"):
        verify_module_ir(model)


def test_weight_wrong_shape_rejected():
    layer = _linear(4, 3)
    layer.weight = torch.randn(3, 5)  # doesn't match in_features
    model = SNNModel(layers=[layer, _lif()])
    with pytest.raises(IRVerificationError, match="weight has shape"):
        verify_module_ir(model)


def test_weight_wrong_rank_rejected():
    layer = _linear(4, 3)
    layer.weight = torch.randn(3, 4, 1)
    model = SNNModel(layers=[layer, _lif()])
    with pytest.raises(IRVerificationError, match="rank"):
        verify_module_ir(model)


def test_bias_wrong_shape_rejected():
    layer = _linear(4, 3)
    layer.bias = torch.randn(5)
    model = SNNModel(layers=[layer, _lif()])
    with pytest.raises(IRVerificationError, match="bias has shape"):
        verify_module_ir(model)


def test_non_positive_dimensions_rejected():
    layer = _linear(4, 3)
    layer.out_features = 0
    model = SNNModel(layers=[layer, _lif()])
    with pytest.raises(IRVerificationError, match="non-positive"):
        verify_module_ir(model)


def test_non_finite_weight_rejected():
    layer = _linear(4, 3)
    layer.weight[0, 0] = float("nan")
    model = SNNModel(layers=[layer, _lif()])
    with pytest.raises(IRVerificationError, match="non-finite"):
        verify_module_ir(model)


def test_beta_wrong_length_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(beta=[0.9, 0.8])])
    with pytest.raises(IRVerificationError, match="beta has length"):
        verify_module_ir(model)


def test_threshold_wrong_length_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(threshold=[1.0, 1.0])])
    with pytest.raises(IRVerificationError, match="threshold has length"):
        verify_module_ir(model)


def test_graded_spikes_factor_wrong_length_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(graded_spikes_factor=[1.0, 1.0])])
    with pytest.raises(IRVerificationError, match="graded_spikes_factor has length"):
        verify_module_ir(model)


def test_per_neuron_length_matching_neuron_count_accepted():
    model = SNNModel(layers=[_linear(4, 3), _lif(beta=[0.9, 0.8, 0.7], threshold=[1.0, 1.1, 1.2])])
    verify_module_ir(model)  # must not raise


def test_unsupported_reset_mechanism_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(reset_mechanism="bogus")])
    with pytest.raises(IRVerificationError, match="reset_mechanism"):
        verify_module_ir(model)


def test_reset_delay_not_bool_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif()])
    model.layers[1].reset_delay = "yes"
    with pytest.raises(IRVerificationError, match="reset_delay"):
        verify_module_ir(model)


def test_inhibition_not_bool_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif()])
    model.layers[1].inhibition = 1
    with pytest.raises(IRVerificationError, match="inhibition"):
        verify_module_ir(model)


def test_empty_state_quant_levels_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(state_quant=StateQuantSpec(levels=()))])
    with pytest.raises(IRVerificationError, match="empty level table"):
        verify_module_ir(model)


def test_non_finite_state_quant_level_rejected():
    model = SNNModel(
        layers=[_linear(4, 3), _lif(state_quant=StateQuantSpec(levels=(0.0, math.inf)))]
    )
    with pytest.raises(IRVerificationError, match="non-finite"):
        verify_module_ir(model)


def test_descending_state_quant_levels_rejected():
    model = SNNModel(
        layers=[_linear(4, 3), _lif(state_quant=StateQuantSpec(levels=(1.0, 0.5)))]
    )
    with pytest.raises(IRVerificationError, match="ascending order"):
        verify_module_ir(model)


def test_duplicate_state_quant_levels_accepted():
    model = SNNModel(
        layers=[_linear(4, 3), _lif(state_quant=StateQuantSpec(levels=(0.0, 0.5, 0.5, 1.0)))]
    )
    verify_module_ir(model)  # duplicates are numerically valid, must not raise


def test_trailing_unpaired_linear_rejected():
    model = SNNModel(layers=[_linear(4, 3), _lif(), _linear(3, 2)])
    with pytest.raises(IRVerificationError, match="must end with a LIF"):
        verify_module_ir(model)
