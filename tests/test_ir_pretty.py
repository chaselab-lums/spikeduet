"""Tests for spikeduet.ir.pretty -- human-readable and
deterministic-dict rendering of the source IR.
"""

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.pretty import format_module_ir, module_ir_to_dict
from tests.test_reference import ChainSNN


def test_format_includes_dimensions_and_bias():
    model = ChainSNN(dims=[(4, 6), (6, 3)], betas=[0.95, 0.9], thresholds=[1.0, 0.8])
    ir = parse_snntorch_model(model)

    text = format_module_ir(ir)

    assert "Linear[0]: 4 -> 6, bias=True" in text
    assert "Linear[1]: 6 -> 3, bias=True" in text
    assert "Leaky[0]: neurons=6" in text
    assert "Leaky[1]: neurons=3" in text
    assert "reset=subtract" in text
    assert "reset_delay=False" in text
    assert "inhibition=False" in text
    assert "state_quant=None" in text


def test_format_never_dumps_raw_weight_values():
    model = ChainSNN(dims=[(4, 6)], betas=[0.95], thresholds=[1.0])
    ir = parse_snntorch_model(model)
    weight_values = model.linears[0].weight.detach().reshape(-1).tolist()

    text = format_module_ir(ir)

    # a distinctive-precision weight value should not appear verbatim
    assert f"{weight_values[0]:.6f}" not in text


def test_format_scalar_vs_per_neuron_beta():
    model = ChainSNN(dims=[(3, 4)], betas=[[0.0, 0.4, 0.8, 1.0]], thresholds=[[0.3, 0.6, 0.9, 1.2]])
    ir = parse_snntorch_model(model)

    text = format_module_ir(ir)

    assert "beta=per_neuron(n=4" in text
    assert "threshold=per_neuron(n=4" in text


def test_to_dict_is_deterministic_and_json_serializable():
    import json

    model = ChainSNN(dims=[(4, 6), (6, 3)], betas=[0.95, 0.9], thresholds=[1.0, 0.8])
    ir = parse_snntorch_model(model)

    d1 = module_ir_to_dict(ir)
    d2 = module_ir_to_dict(ir)
    assert d1 == d2
    json.dumps(d1)  # must not raise

    assert d1["layers"][0] == {"kind": "Linear", "in_features": 4, "out_features": 6, "bias": True}
    assert d1["layers"][1]["kind"] == "Leaky"
    assert d1["layers"][1]["neurons"] == 6
    assert d1["layers"][1]["beta_len"] == 1


def test_to_dict_excludes_weight_values():
    model = ChainSNN(dims=[(4, 6)], betas=[0.95], thresholds=[1.0])
    ir = parse_snntorch_model(model)

    d = module_ir_to_dict(ir)

    for layer in d["layers"]:
        assert "weight" not in layer
        assert "bias_values" not in layer
