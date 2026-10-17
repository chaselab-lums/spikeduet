"""Tests for spikeduet.backend.model_header -- the Python -> C model-data export boundary.

Uses the same synthetic-model style as tests/test_frontend.py and
tests/test_reference.py (ChainSNN, reused from tests/test_reference.py):
build a snnTorch model, parse it with Frontend V1, generate the C header,
and check the generated text preserves every value exactly and generically
(no reliance on module names like "fc1"/"fc2").
"""

import re

import pytest
import torch
import torch.nn as nn
import snntorch as snn

from spikeduet.backend.model_header import generate_c_model
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from tests.test_reference import ChainSNN


def _extract_array(header: str, array_name: str):
    match = re.search(
        rf"\b{re.escape(array_name)}\[\d+\] = \{{(.*?)\}};", header, re.DOTALL
    )
    assert match, f"array '{array_name}' not found in generated header"
    body = match.group(1).strip()
    if not body:
        return []
    return [float(tok.strip().rstrip("f")) for tok in body.split(",") if tok.strip()]


def _extract_macro(header: str, name: str) -> int:
    match = re.search(rf"#define {re.escape(name)} (\d+)", header)
    assert match, f"macro '{name}' not found"
    return int(match.group(1))


def test_two_layer_export_preserves_values_and_dims():
    model = ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.95, 0.9], thresholds=[1.0, 0.8])
    ir = parse_snntorch_model(model)

    header = generate_c_model(ir)

    assert _extract_macro(header, "ISCAS_INPUT_DIM") == 4
    assert _extract_macro(header, "ISCAS_OUTPUT_DIM") == 2
    assert _extract_macro(header, "ISCAS_NUM_LAYERS") == 2

    w0 = _extract_array(header, "iscas_layer0_weight")
    b0 = _extract_array(header, "iscas_layer0_bias")
    w1 = _extract_array(header, "iscas_layer1_weight")
    b1 = _extract_array(header, "iscas_layer1_bias")

    expected_w0 = model.linears[0].weight.detach().reshape(-1).tolist()
    expected_b0 = model.linears[0].bias.detach().reshape(-1).tolist()
    expected_w1 = model.linears[1].weight.detach().reshape(-1).tolist()
    expected_b1 = model.linears[1].bias.detach().reshape(-1).tolist()

    assert w0 == pytest.approx(expected_w0, abs=1e-6)
    assert b0 == pytest.approx(expected_b0, abs=1e-6)
    assert w1 == pytest.approx(expected_w1, abs=1e-6)
    assert b1 == pytest.approx(expected_b1, abs=1e-6)

    assert (
        "{ 4, 3, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )
    assert (
        "{ 3, 2, iscas_layer1_weight, iscas_layer1_bias, "
        "iscas_layer1_beta, 1, iscas_layer1_threshold, 1, "
        "iscas_layer1_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )
    assert _extract_array(header, "iscas_layer0_beta") == [0.95]
    assert _extract_array(header, "iscas_layer0_threshold") == [1.0]
    assert _extract_array(header, "iscas_layer1_beta") == [0.9]
    assert _extract_array(header, "iscas_layer1_threshold") == pytest.approx([0.8])
    assert _extract_array(header, "iscas_layer0_graded_spikes_factor") == [1.0]
    assert _extract_array(header, "iscas_layer1_graded_spikes_factor") == [1.0]


def test_three_layer_export_layer_ordering_matches_ir():
    model = ChainSNN(
        dims=[(4, 6), (6, 4), (4, 2)],
        betas=[0.95, 0.90, 0.92],
        thresholds=[1.0, 0.8, 1.1],
    )
    ir = parse_snntorch_model(model)

    header = generate_c_model(ir)

    assert _extract_macro(header, "ISCAS_NUM_LAYERS") == 3

    # descriptor table order must match IR order (fc1,lif1,fc2,lif2,fc3,lif3)
    table_match = re.search(
        r"ISCAS_LAYERS\[ISCAS_NUM_LAYERS\] = \{(.*?)\};", header, re.DOTALL
    )
    assert table_match
    rows = [line.strip() for line in table_match.group(1).strip().splitlines() if line.strip()]
    assert len(rows) == 3
    assert rows[0].startswith("{ 4, 6,")
    assert rows[1].startswith("{ 6, 4,")
    assert rows[2].startswith("{ 4, 2,")
    for i, row in enumerate(rows):
        assert f"iscas_layer{i}_weight" in row


def test_bias_false_layer_emits_null_and_no_bias_array():
    model = ChainSNN(dims=[(5, 3)], betas=[0.9], thresholds=[1.0], biases=[False])
    ir = parse_snntorch_model(model)

    header = generate_c_model(ir)

    assert "iscas_layer0_bias" not in header
    assert (
        "{ 5, 3, iscas_layer0_weight, NULL, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )


def test_arbitrary_dimensions_are_exported_correctly():
    model = ChainSNN(dims=[(17, 13), (13, 9)], betas=[0.8, 0.99], thresholds=[0.5, 2.5])
    ir = parse_snntorch_model(model)

    header = generate_c_model(ir)

    assert _extract_macro(header, "ISCAS_INPUT_DIM") == 17
    assert _extract_macro(header, "ISCAS_OUTPUT_DIM") == 9
    assert len(_extract_array(header, "iscas_layer0_weight")) == 13 * 17
    assert len(_extract_array(header, "iscas_layer1_weight")) == 9 * 13


def test_export_does_not_rely_on_module_names():
    class OddlyNamedSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(6, 4)
            self.spiking_stage = snn.Leaky(
                beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False
            )

        def forward(self, x):
            mem = self.spiking_stage.reset_mem()
            cur = self.input_projection(x)
            spk, mem = self.spiking_stage(cur, mem)
            return spk, mem

    model = OddlyNamedSNN()
    ir = parse_snntorch_model(model)

    header = generate_c_model(ir)

    # generated identifiers are purely positional (iscas_layerN_*), never
    # derived from the source module's attribute names
    assert "input_projection" not in header
    assert "spiking_stage" not in header
    assert "iscas_layer0_weight" in header


def test_export_is_deterministic():
    model = ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.95, 0.9], thresholds=[1.0, 0.8])
    ir = parse_snntorch_model(model)

    header_a = generate_c_model(ir)
    header_b = generate_c_model(ir)

    assert header_a == header_b

    # re-parsing the same untouched model must also yield identical output
    ir_again = parse_snntorch_model(model)
    header_c = generate_c_model(ir_again)
    assert header_a == header_c


def test_weights_are_flattened_row_major_out_by_in():
    """Weights are emitted row-major ``[out_features][in_features]``, i.e. ``weight.reshape(-1)``
    in order, with no transpose."""
    torch.manual_seed(0)
    model = ChainSNN(dims=[(45, 6), (6, 3)], betas=[0.95, 0.95], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert _extract_macro(header, "ISCAS_INPUT_DIM") == 45
    assert _extract_macro(header, "ISCAS_OUTPUT_DIM") == 3

    w1 = model.linears[0].weight.detach().numpy()  # shape (6, 45): [out][in]
    row_major_flat = []
    for row in w1:
        row_major_flat.extend(float(x) for x in row)

    generated_flat = _extract_array(header, "iscas_layer0_weight")
    assert generated_flat == pytest.approx(row_major_flat, abs=1e-6)

    w2 = model.linears[1].weight.detach().numpy()  # shape (3, 6): [out][in]
    row_major_flat_2 = []
    for row in w2:
        row_major_flat_2.extend(float(x) for x in row)
    generated_flat_2 = _extract_array(header, "iscas_layer1_weight")
    assert generated_flat_2 == pytest.approx(row_major_flat_2, abs=1e-6)


def test_reset_mechanism_and_delay_are_emitted_generically():
    """Non-default reset_mechanism/reset_delay values flow through to
    the generated descriptor table -- 0=subtract/1=zero/2=none matching
    snntorch.SpikingNeuron.reset_dict, reset_delay as 0/1."""
    model = ChainSNN(
        dims=[(3, 2), (2, 2), (2, 2)],
        betas=[0.9, 0.9, 0.9],
        thresholds=[1.0, 1.0, 1.0],
        reset_mechanisms=["subtract", "zero", "none"],
        reset_delays=[False, True, False],
    )
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert (
        "{ 3, 2, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )
    assert (
        "{ 2, 2, iscas_layer1_weight, iscas_layer1_bias, "
        "iscas_layer1_beta, 1, iscas_layer1_threshold, 1, "
        "iscas_layer1_graded_spikes_factor, 1, 1, 1, 0, NULL, 0 }" in header
    )
    assert (
        "{ 2, 2, iscas_layer2_weight, iscas_layer2_bias, "
        "iscas_layer2_beta, 1, iscas_layer2_threshold, 1, "
        "iscas_layer2_graded_spikes_factor, 1, 2, 0, 0, NULL, 0 }" in header
    )


def test_per_neuron_beta_and_threshold_are_emitted_as_arrays():
    """Per-neuron beta/threshold flow through to a length-N array with
    beta_len/threshold_len == N, not a scalar."""
    model = ChainSNN(
        dims=[(3, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
    )
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert _extract_array(header, "iscas_layer0_beta") == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert _extract_array(header, "iscas_layer0_threshold") == pytest.approx([0.3, 0.6, 0.9, 1.2])
    assert (
        "{ 3, 4, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 4, iscas_layer0_threshold, 4, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )


def test_scalar_beta_emits_single_element_array_not_n_copies():
    """A scalar layer's beta/threshold arrays
    must have exactly 1 element, never `out_features` repeated copies."""
    model = ChainSNN(dims=[(5, 50)], betas=[0.9], thresholds=[1.0])
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert len(_extract_array(header, "iscas_layer0_beta")) == 1
    assert len(_extract_array(header, "iscas_layer0_threshold")) == 1


def test_graded_spikes_factor_is_emitted_as_array():
    """Per-neuron graded_spikes_factor flows through to a length-N
    array with graded_spikes_factor_len == N, not a scalar, like
    beta and threshold."""
    model = ChainSNN(
        dims=[(3, 4)],
        betas=[0.9],
        thresholds=[1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5]],
    )
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert _extract_array(header, "iscas_layer0_graded_spikes_factor") == pytest.approx(
        [0.25, 0.75, 1.5, 2.5]
    )
    assert (
        "{ 3, 4, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 4, 0, 0, 0, NULL, 0 }" in header
    )


def test_scalar_graded_spikes_factor_emits_single_element_array():
    """A scalar graded_spikes_factor emits a
    1-element array, never `out_features` repeated copies."""
    model = ChainSNN(dims=[(5, 50)], betas=[0.9], thresholds=[1.0], graded_spikes_factors=[2.5])
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert len(_extract_array(header, "iscas_layer0_graded_spikes_factor")) == 1


def test_inhibition_is_emitted_generically():
    """inhibition=True flows through to the generated descriptor
    table as a trailing 0/1 int field, per-layer, not special-cased."""
    model = ChainSNN(
        dims=[(3, 2), (2, 2)],
        betas=[0.9, 0.9],
        thresholds=[1.0, 1.0],
        inhibitions=[True, False],
    )
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    assert (
        "{ 3, 2, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 1, NULL, 0 }" in header
    )
    assert (
        "{ 2, 2, iscas_layer1_weight, iscas_layer1_bias, "
        "iscas_layer1_beta, 1, iscas_layer1_threshold, 1, "
        "iscas_layer1_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )


def test_state_quant_is_emitted_as_level_table():
    """A recognized built-in state_quant flows through to an emitted
    iscas_layerN_state_quant_levels[] array plus a state_quant_num_levels
    field, mirroring the beta/threshold/graded_spikes_factor pointer+
    length convention (but always per-layer, never per-neuron)."""
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0)
    model = ChainSNN(
        dims=[(3, 2), (2, 2)],
        betas=[0.9, 0.9],
        thresholds=[1.0, 1.0],
        state_quants=[q, False],
    )
    ir = parse_snntorch_model(model)
    header = generate_c_model(ir)

    levels = _extract_array(header, "iscas_layer0_state_quant_levels")
    assert len(levels) == 4  # 2**2
    assert (
        "{ 3, 2, iscas_layer0_weight, iscas_layer0_bias, "
        "iscas_layer0_beta, 1, iscas_layer0_threshold, 1, "
        "iscas_layer0_graded_spikes_factor, 1, 0, 0, 0, "
        "iscas_layer0_state_quant_levels, 4 }" in header
    )
    # disabled state_quant on layer 1: NULL, 0, no array emitted for it
    assert "iscas_layer1_state_quant_levels" not in header
    assert (
        "{ 2, 2, iscas_layer1_weight, iscas_layer1_bias, "
        "iscas_layer1_beta, 1, iscas_layer1_threshold, 1, "
        "iscas_layer1_graded_spikes_factor, 1, 0, 0, 0, NULL, 0 }" in header
    )
