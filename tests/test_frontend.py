"""Tests for spikeduet.frontend.snntorch_parser using synthetic snnTorch models.

These models are synthetic stand-ins built purely to stress the parser;
no trained checkpoint is required or used here. Frontend V1 is FX-based
(see spikeduet/frontend/snntorch_parser.py and notebooks/00_iscas_development.ipynb),
so every model below needs a real forward() with explicit membrane-state
handling and tuple unpacking -- the exact pattern the parser validates.
"""

import pytest
import torch
import torch.nn as nn
import snntorch as snn

from spikeduet.frontend.snntorch_parser import (
    UnsupportedGraphStructureError,
    UnsupportedLeakyConfigurationError,
    UnsupportedOperationError,
    parse_snntorch_model,
)
from spikeduet.ir.model import LIFLayer, LinearLayer


class TinyTwoLayerSNN(nn.Module):
    """Linear(45,6) -> LIF -> Linear(6,3) -> LIF."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc2 = nn.Linear(6, 3)
        self.lif2 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem2


class DeeperSNN(nn.Module):
    """Linear(45,16) -> LIF -> Linear(16,8) -> LIF -> Linear(8,3) -> LIF."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 16)
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc2 = nn.Linear(16, 8)
        self.lif2 = snn.Leaky(beta=0.90, threshold=0.8, reset_mechanism="subtract", reset_delay=False)
        self.fc3 = nn.Linear(8, 3)
        self.lif3 = snn.Leaky(beta=0.92, threshold=1.1, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem3


class ArbitraryNamedSNN(nn.Module):
    """Same shape as TinyTwoLayerSNN, but with names unrelated to fc1/lif1/etc."""

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(45, 6)
        self.spiking_stage_alpha = snn.Leaky(
            beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False
        )
        self.output_projection = nn.Linear(6, 3)
        self.spiking_stage_beta = snn.Leaky(
            beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False
        )

    def forward(self, x):
        mem_a = self.spiking_stage_alpha.init_leaky()
        mem_b = self.spiking_stage_beta.init_leaky()
        cur1 = self.input_projection(x)
        spk1, mem_a = self.spiking_stage_alpha(cur1, mem_a)
        cur2 = self.output_projection(spk1)
        spk2, mem_b = self.spiking_stage_beta(cur2, mem_b)
        return spk2, mem_b


class SequentialSNN(nn.Sequential):
    """An nn.Sequential SNN with a custom forward (Sequential's own auto-chain
    forward only ever passes a single positional value between children, which
    cannot carry the explicit (cur, mem) / (spk, mem) protocol Leaky needs)."""

    def __init__(self):
        super().__init__(
            nn.Linear(45, 6),
            snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False),
            nn.Linear(6, 3),
            snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False),
        )

    def forward(self, x):
        fc1, lif1, fc2, lif2 = self
        mem1 = lif1.init_leaky()
        mem2 = lif2.init_leaky()
        cur1 = fc1(x)
        spk1, mem1 = lif1(cur1, mem1)
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)
        return spk2, mem2


class NoBiasSNN(nn.Module):
    """Linear(45,6, bias=False) -> LIF."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6, bias=False)
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class UnsupportedOpSNN(nn.Module):
    """A Linear -> LIF chain with an unsupported nn.ReLU inserted between them."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.relu = nn.ReLU()
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        act1 = self.relu(cur1)
        spk1, mem1 = self.lif1(act1, mem1)
        return spk1, mem1


class ReorderedRegistrationSNN(nn.Module):
    """Registration order (__init__) deliberately differs from forward()
    execution order: lif1 and lif2 are assigned before the fc layers they
    actually consume. named_modules() order here is lif1, fc1, lif2, fc2 --
    not a valid [Linear -> Leaky]^N reading -- while true forward() order is
    fc1, lif1, fc2, lif2, which is valid. Proves the parser follows FX
    execution order, not registration order."""

    def __init__(self):
        super().__init__()
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc1 = nn.Linear(45, 6)
        self.lif2 = snn.Leaky(beta=0.90, threshold=0.8, reset_mechanism="subtract", reset_delay=False)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem2


class ReusedLinearSNN(nn.Module):
    """Reuses fc_shared for both projections -- must be rejected."""

    def __init__(self):
        super().__init__()
        self.fc_shared = nn.Linear(4, 4)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur1 = self.fc_shared(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc_shared(spk1)  # reuse: same module called twice
        spk2, mem2 = self.lif2(cur2, mem2)
        return spk2, mem2


class BranchingSNN(nn.Module):
    """Two parallel LIF branches fed from the same spike output, merged by
    addition -- must be rejected (not a single sequential chain)."""

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(4, 4)
        self.lif_in = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc_branch_a = nn.Linear(4, 2)
        self.fc_branch_b = nn.Linear(4, 2)
        self.lif_a = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.lif_b = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem_in = self.lif_in.init_leaky()
        mem_a = self.lif_a.init_leaky()
        mem_b = self.lif_b.init_leaky()

        cur_in = self.fc_in(x)
        spk_in, mem_in = self.lif_in(cur_in, mem_in)

        cur_a = self.fc_branch_a(spk_in)
        spk_a, mem_a = self.lif_a(cur_a, mem_a)

        cur_b = self.fc_branch_b(spk_in)
        spk_b, mem_b = self.lif_b(cur_b, mem_b)

        return spk_a + spk_b


class UnsupportedCallFunctionSNN(nn.Module):
    """Inserts arithmetic (a call_function) between Linear and Leaky."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        cur1 = cur1 * 2.0  # unsupported call_function between layers
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


def test_tiny_two_layer_snn():
    model = TinyTwoLayerSNN()
    parsed = parse_snntorch_model(model)

    assert [type(layer) for layer in parsed.layers] == [
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
    ]

    fc1, lif1, fc2, lif2 = parsed.layers

    assert fc1.in_features == 45
    assert fc1.out_features == 6
    assert fc1.weight.shape == (6, 45)
    assert fc1.bias is not None
    assert fc1.bias.shape == (6,)
    assert fc1.weight.device.type == "cpu"
    assert not fc1.weight.requires_grad

    assert lif1.beta == pytest.approx([0.95])
    assert lif1.threshold == pytest.approx([1.0])
    assert lif1.reset_mechanism == "subtract"

    assert fc2.in_features == 6
    assert fc2.out_features == 3

    assert lif2.beta == pytest.approx([0.95])


def test_deeper_snn_yields_expected_order_and_params():
    model = DeeperSNN()
    parsed = parse_snntorch_model(model)

    assert [type(layer) for layer in parsed.layers] == [
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
    ]

    dims = [(l.in_features, l.out_features) for l in parsed.layers if isinstance(l, LinearLayer)]
    assert dims == [(45, 16), (16, 8), (8, 3)]

    betas = [l.beta[0] for l in parsed.layers if isinstance(l, LIFLayer)]
    assert betas == pytest.approx([0.95, 0.90, 0.92])

    thresholds = [l.threshold[0] for l in parsed.layers if isinstance(l, LIFLayer)]
    assert thresholds == pytest.approx([1.0, 0.8, 1.1])


def test_arbitrary_module_names_are_not_relied_upon():
    model = ArbitraryNamedSNN()
    parsed = parse_snntorch_model(model)

    assert [type(layer) for layer in parsed.layers] == [
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
    ]
    # Names are preserved as discovered, not normalized to fc1/lif1/etc.
    assert parsed.layers[0].name == "input_projection"
    assert parsed.layers[1].name == "spiking_stage_alpha"
    assert parsed.layers[2].name == "output_projection"
    assert parsed.layers[3].name == "spiking_stage_beta"


def test_nn_sequential_model_is_parsed_correctly():
    model = SequentialSNN()
    parsed = parse_snntorch_model(model)

    assert [type(layer) for layer in parsed.layers] == [
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
    ]
    assert parsed.layers[0].in_features == 45
    assert parsed.layers[2].out_features == 3


def test_linear_layer_without_bias():
    model = NoBiasSNN()
    parsed = parse_snntorch_model(model)

    fc1 = parsed.layers[0]
    assert isinstance(fc1, LinearLayer)
    assert fc1.bias is None


def test_unsupported_operation_raises_clear_error():
    model = UnsupportedOpSNN()

    with pytest.raises(UnsupportedOperationError) as excinfo:
        parse_snntorch_model(model)

    assert excinfo.value.module_name == "relu"
    assert excinfo.value.module_type is nn.ReLU
    assert "relu" in str(excinfo.value)
    assert "ReLU" in str(excinfo.value)


def test_parser_follows_execution_order_not_registration_order():
    """named_modules() order for this model is lif1, fc1, lif2, fc2 -- which
    is not a valid Linear-first reading. True forward() execution order is
    fc1, lif1, fc2, lif2, which is valid. The parser must produce the
    latter."""
    model = ReorderedRegistrationSNN()

    registration_order = [name for name, _ in model.named_modules() if name]
    assert registration_order == ["lif1", "fc1", "lif2", "fc2"]

    parsed = parse_snntorch_model(model)

    assert [layer.name for layer in parsed.layers] == ["fc1", "lif1", "fc2", "lif2"]
    assert [type(layer) for layer in parsed.layers] == [
        LinearLayer,
        LIFLayer,
        LinearLayer,
        LIFLayer,
    ]


def test_semantic_module_reuse_is_rejected():
    model = ReusedLinearSNN()

    with pytest.raises(UnsupportedGraphStructureError, match="reuse"):
        parse_snntorch_model(model)


def test_branching_is_rejected():
    model = BranchingSNN()

    with pytest.raises(UnsupportedGraphStructureError, match="consumer"):
        parse_snntorch_model(model)


def test_unsupported_call_function_is_rejected():
    model = UnsupportedCallFunctionSNN()

    with pytest.raises(UnsupportedGraphStructureError):
        parse_snntorch_model(model)


def test_parsing_does_not_mutate_leaky_membrane_state():
    model = DeeperSNN()
    mem_before = {
        name: getattr(model, name).mem.clone()
        for name in ("lif1", "lif2", "lif3")
    }

    parse_snntorch_model(model)

    for name, before in mem_before.items():
        after = getattr(model, name).mem
        assert type(after) is type(before)
        assert after.shape == before.shape
        assert (after == before).all()


class DefaultResetDelaySNN(nn.Module):
    """Leaky constructed WITHOUT reset_delay -- snnTorch defaults this to
    True, which delays the reset by
    one step relative to the threshold crossing. Both reset_delay values are
    supported and preserved in the IR."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.lif1 = snn.Leaky(beta=0.95, threshold=1.0, reset_mechanism="subtract")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class ZeroResetMechanismSNN(nn.Module):
    """reset_mechanism='zero' (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.lif1 = snn.Leaky(
            beta=0.95, threshold=1.0, reset_mechanism="zero", reset_delay=False
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class NoneResetMechanismSNN(nn.Module):
    """reset_mechanism='none' (pure integration, never reset) (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(45, 6)
        self.lif1 = snn.Leaky(
            beta=0.95, threshold=1.0, reset_mechanism="none", reset_delay=False
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class UnknownResetMechanismSNN(nn.Module):
    """An invalid reset_mechanism string. snnTorch itself validates this at
    Leaky construction time (ValueError), so the frontend's own rejection
    path (UnsupportedLeakyConfigurationError for any string outside
    {"subtract","zero","none"}) is unreachable through a validly
    constructed Leaky module -- included anyway for defense in depth,
    constructing the module directly and patching the buffer afterward to
    bypass snnTorch's own validation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract")
        object.__setattr__(self.lif1, "_reset_mechanism", "bogus")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class InhibitionSNN(nn.Module):
    """inhibition=True (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False, inhibition=True,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class GradedSpikesScalarSNN(nn.Module):
    """A scalar graded_spikes_factor != 1.0 (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False, graded_spikes_factor=2.5,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class GradedSpikesPerNeuronSNN(nn.Module):
    """A per-neuron graded_spikes_factor whose length exactly matches
    the layer's neuron count (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False,
            graded_spikes_factor=torch.tensor([0.25, 0.75, 1.5]),
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class WrongLengthGradedSpikesSNN(nn.Module):
    """A vector graded_spikes_factor whose length is neither 1 nor the
    layer's neuron count (3) -- must be rejected loudly."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False,
            graded_spikes_factor=torch.tensor([0.5, 0.5]),  # 2 values, but 3 neurons
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class LearnedScalarGradedSpikesSNN(nn.Module):
    """learn_graded_spikes_factor=True with a scalar initial value.
    Must produce the SAME IR as a fixed scalar with the identical
    resolved value."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False, graded_spikes_factor=1.7,
            learn_graded_spikes_factor=True,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class ArbitraryCallableStateQuantSNN(nn.Module):
    """An arbitrary (non-snnTorch-built-in) state_quant callable --
    must still be rejected loudly, not silently ignored. Backend/IR V1
    represents quantization as a resolved level table, never an
    executable Python function."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False, state_quant=lambda mem: torch.round(mem),
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class BuiltinStateQuantSNN(nn.Module):
    """state_quant produced by snntorch.functional.quant.state_quant(...) (supported)."""

    def __init__(self):
        super().__init__()
        from snntorch.functional import quant

        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False,
            state_quant=quant.state_quant(num_bits=2, uniform=True, threshold=1.0),
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class BuiltinNonUniformStateQuantSNN(nn.Module):
    """A non-uniform (thr_centered) built-in quantizer (supported), exercising the other branch of
    snntorch.functional.quant.state_quant's level construction."""

    def __init__(self):
        super().__init__()
        from snntorch.functional import quant

        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9, threshold=1.0, reset_mechanism="subtract",
            reset_delay=False,
            state_quant=quant.state_quant(
                num_bits=3, uniform=False, thr_centered=True, threshold=1.0
            ),
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class PerNeuronBetaSNN(nn.Module):
    """A per-neuron (vector) beta whose length exactly matches the
    layer's neuron count (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=torch.tensor([0.9, 0.8, 0.7]),
            threshold=1.0,
            reset_mechanism="subtract",
            reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class PerNeuronThresholdSNN(nn.Module):
    """A per-neuron (vector) threshold whose length exactly matches the
    layer's neuron count (supported)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=torch.tensor([1.0, 0.5, 2.0]),
            reset_mechanism="subtract",
            reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class WrongLengthBetaSNN(nn.Module):
    """A vector beta whose length is neither 1 nor the layer's neuron
    count (3) -- must be rejected loudly, not silently truncated/broadcast."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=torch.tensor([0.9, 0.8]),  # 2 values, but 3 neurons
            threshold=1.0,
            reset_mechanism="subtract",
            reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class WrongLengthThresholdSNN(nn.Module):
    """A vector threshold whose length is neither 1 nor the layer's
    neuron count (3) -- must be rejected loudly."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=torch.tensor([1.0, 0.5, 2.0, 0.3]),  # 4 values, but 3 neurons
            reset_mechanism="subtract",
            reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class LearnedScalarBetaSNN(nn.Module):
    """learn_beta=True with a scalar initial value. Must produce the
    SAME IR beta as a fixed scalar with the identical resolved value --
    the compiler never inspects learn_beta, only the live module.beta."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=0.85, threshold=1.0, learn_beta=True,
            reset_mechanism="subtract", reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


class LearnedPerNeuronBetaSNN(nn.Module):
    """learn_beta=True with a per-neuron initial value. Must produce
    the SAME IR beta as a fixed per-neuron tensor with the identical
    resolved values."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.lif1 = snn.Leaky(
            beta=torch.tensor([0.9, 0.8, 0.7]), threshold=1.0, learn_beta=True,
            reset_mechanism="subtract", reset_delay=False,
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        return spk1, mem1


def test_default_reset_delay_is_accepted():
    """reset_delay=True (snnTorch's own default) is accepted and
    captured in the IR, not silently rewritten or rejected."""
    model = DefaultResetDelaySNN()
    assert model.lif1.reset_delay is True  # confirm snnTorch's actual default

    ir = parse_snntorch_model(model)
    assert ir.layers[1].reset_delay is True
    assert ir.layers[1].reset_mechanism == "subtract"


def test_zero_reset_mechanism_is_accepted():
    """reset_mechanism='zero' is accepted and captured in the IR."""
    model = ZeroResetMechanismSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].reset_mechanism == "zero"
    assert ir.layers[1].reset_delay is False


def test_none_reset_mechanism_is_accepted():
    """reset_mechanism='none' is accepted and captured in the IR."""
    model = NoneResetMechanismSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].reset_mechanism == "none"


def test_unknown_reset_mechanism_string_is_rejected():
    model = UnknownResetMechanismSNN()

    with pytest.raises(UnsupportedLeakyConfigurationError, match="reset_mechanism"):
        parse_snntorch_model(model)


def test_ordinary_leaky_still_parses():
    """The plain configuration (subtract + immediate reset, scalar beta/threshold) parses with
    graded_spikes_factor defaulting to [1.0] (snnTorch's own default, unscaled)."""
    model = TinyTwoLayerSNN()
    ir = parse_snntorch_model(model)
    assert len(ir.layers) == 4
    assert ir.layers[1].reset_mechanism == "subtract"
    assert ir.layers[1].reset_delay is False
    assert ir.layers[1].graded_spikes_factor == pytest.approx([1.0])
    assert ir.layers[3].reset_mechanism == "subtract"
    assert ir.layers[3].reset_delay is False
    assert ir.layers[3].graded_spikes_factor == pytest.approx([1.0])


def test_inhibition_is_accepted():
    """inhibition=True is accepted and captured in the IR."""
    model = InhibitionSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].inhibition is True


def test_inhibition_defaults_to_false():
    """Regression guard: an ordinary (non-inhibited) model must still
    parse with inhibition=False, not silently flipped."""
    model = TinyTwoLayerSNN()
    ir = parse_snntorch_model(model)
    assert ir.layers[1].inhibition is False
    assert ir.layers[3].inhibition is False


def test_graded_spikes_factor_scalar_is_accepted():
    """A scalar graded_spikes_factor != 1.0 is accepted and
    captured in the IR."""
    model = GradedSpikesScalarSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].graded_spikes_factor == pytest.approx([2.5])


def test_arbitrary_state_quant_callable_is_rejected():
    """An arbitrary (non-built-in) state_quant callable must still
    fail loudly, not be silently ignored -- Backend/IR V1 only recognizes
    snntorch.functional.quant.state_quant(...) closures."""
    model = ArbitraryCallableStateQuantSNN()

    with pytest.raises(UnsupportedLeakyConfigurationError, match="state_quant"):
        parse_snntorch_model(model)


def test_builtin_uniform_state_quant_is_accepted():
    """A recognized snntorch.functional.quant.state_quant(...) closure
    (uniform) is accepted and its resolved levels captured in the IR."""
    from snntorch.functional import quant

    model = BuiltinStateQuantSNN()
    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0)
    expected_levels = tuple(
        q.__closure__[q.__code__.co_freevars.index("levels")].cell_contents.tolist()
    )

    ir = parse_snntorch_model(model)
    assert ir.layers[1].state_quant is not None
    assert ir.layers[1].state_quant.levels == pytest.approx(expected_levels)


def test_builtin_nonuniform_state_quant_is_accepted():
    """A recognized non-uniform (thr_centered) built-in quantizer is
    also accepted."""
    model = BuiltinNonUniformStateQuantSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].state_quant is not None
    assert len(ir.layers[1].state_quant.levels) == 8  # 2**3 levels for num_bits=3
    levels = ir.layers[1].state_quant.levels
    assert list(levels) == sorted(levels)  # verified ascending order (see StateQuantSpec)


def test_state_quant_disabled_is_none_in_ir():
    """Regression guard: an ordinary (state_quant=False) model must
    still parse with state_quant=None, not silently populated."""
    model = TinyTwoLayerSNN()
    ir = parse_snntorch_model(model)
    assert ir.layers[1].state_quant is None
    assert ir.layers[3].state_quant is None


def test_per_neuron_beta_is_accepted():
    """A beta of length == neuron count is accepted and captured
    element-for-element in IR order."""
    model = PerNeuronBetaSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].beta == pytest.approx([0.9, 0.8, 0.7])


def test_per_neuron_threshold_is_accepted():
    """A threshold of length == neuron count is accepted and captured
    element-for-element in IR order."""
    model = PerNeuronThresholdSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].threshold == pytest.approx([1.0, 0.5, 2.0])


def test_wrong_length_beta_is_rejected():
    """A beta whose length is neither 1 nor the neuron count must be
    rejected loudly, not silently truncated/padded/broadcast."""
    model = WrongLengthBetaSNN()

    with pytest.raises(UnsupportedLeakyConfigurationError, match="beta"):
        parse_snntorch_model(model)


def test_wrong_length_threshold_is_rejected():
    """A threshold whose length is neither 1 nor the neuron count must
    be rejected loudly."""
    model = WrongLengthThresholdSNN()

    with pytest.raises(UnsupportedLeakyConfigurationError, match="threshold"):
        parse_snntorch_model(model)


def test_learned_scalar_beta_matches_fixed_scalar_beta():
    """learn_beta=True must produce the identical IR representation as
    a fixed scalar with the same resolved value -- the compiler extracts
    the final live value regardless of how it was obtained."""
    fixed_model = TinyTwoLayerSNN()
    with torch.no_grad():
        fixed_model.lif1.beta.copy_(torch.tensor(0.85))

    learned_model = LearnedScalarBetaSNN()
    assert isinstance(learned_model.lif1.beta, torch.nn.Parameter)  # confirm it really is learned

    fixed_ir = parse_snntorch_model(fixed_model)
    learned_ir = parse_snntorch_model(learned_model)

    assert fixed_ir.layers[1].beta == pytest.approx(learned_ir.layers[1].beta)
    assert learned_ir.layers[1].beta == pytest.approx([0.85])


def test_learned_per_neuron_beta_matches_fixed_per_neuron_beta():
    """Same equivalence, for a per-neuron learned beta."""
    fixed_model = PerNeuronBetaSNN()
    learned_model = LearnedPerNeuronBetaSNN()
    assert isinstance(learned_model.lif1.beta, torch.nn.Parameter)

    fixed_ir = parse_snntorch_model(fixed_model)
    learned_ir = parse_snntorch_model(learned_model)

    assert fixed_ir.layers[1].beta == pytest.approx(learned_ir.layers[1].beta)
    assert learned_ir.layers[1].beta == pytest.approx([0.9, 0.8, 0.7])


def test_graded_spikes_factor_per_neuron_is_accepted():
    """A graded_spikes_factor of length == neuron count is accepted
    and captured element-for-element in IR order."""
    model = GradedSpikesPerNeuronSNN()

    ir = parse_snntorch_model(model)
    assert ir.layers[1].graded_spikes_factor == pytest.approx([0.25, 0.75, 1.5])


def test_wrong_length_graded_spikes_factor_is_rejected():
    """A graded_spikes_factor whose length is neither 1 nor the neuron
    count must be rejected loudly, not silently truncated/padded/broadcast."""
    model = WrongLengthGradedSpikesSNN()

    with pytest.raises(UnsupportedLeakyConfigurationError, match="graded_spikes_factor"):
        parse_snntorch_model(model)


def test_learned_scalar_graded_spikes_factor_matches_fixed():
    """learn_graded_spikes_factor=True must produce the identical IR
    representation as a fixed scalar with the same resolved value."""
    fixed_model = GradedSpikesScalarSNN()
    with torch.no_grad():
        fixed_model.lif1.graded_spikes_factor.copy_(torch.tensor(1.7))

    learned_model = LearnedScalarGradedSpikesSNN()
    assert isinstance(learned_model.lif1.graded_spikes_factor, torch.nn.Parameter)

    fixed_ir = parse_snntorch_model(fixed_model)
    learned_ir = parse_snntorch_model(learned_model)

    assert fixed_ir.layers[1].graded_spikes_factor == pytest.approx(
        learned_ir.layers[1].graded_spikes_factor
    )
    assert learned_ir.layers[1].graded_spikes_factor == pytest.approx([1.7])
