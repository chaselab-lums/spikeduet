"""Equivalence tests: spikeduet.reference.simulator vs. real snnTorch models.

For each model: parse a *fresh, never-forward-called* instance into IR
(parsing itself never executes the model -- see
spikeduet/frontend/snntorch_parser.py -- so this is safe), run the
independent IR executor over an input sequence, and separately drive an
independently deep-copied snnTorch instance (with identical parameters)
over the same sequence, resetting its membrane state explicitly first.
Every per-layer current/spike/membrane value is compared at every
timestep, not just the final output.
"""

import copy

import pytest
import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import quant

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.reference.simulator import initialize_state, run_sequence, step, _quantize_state

ATOL = 1e-5


class ChainSNN(nn.Module):
    """A generic [Linear -> Leaky]^N model of the Frontend V1 supported form."""

    def __init__(
        self,
        dims,
        betas,
        thresholds,
        biases=None,
        reset_mechanisms=None,
        reset_delays=None,
        graded_spikes_factors=None,
        inhibitions=None,
        state_quants=None,
    ):
        super().__init__()
        n = len(dims)
        biases = biases if biases is not None else [True] * n
        reset_mechanisms = reset_mechanisms if reset_mechanisms is not None else ["subtract"] * n
        reset_delays = reset_delays if reset_delays is not None else [False] * n
        graded_spikes_factors = (
            graded_spikes_factors if graded_spikes_factors is not None else [1.0] * n
        )
        inhibitions = inhibitions if inhibitions is not None else [False] * n
        state_quants = state_quants if state_quants is not None else [False] * n
        self.linears = nn.ModuleList(
            nn.Linear(in_f, out_f, bias=b) for (in_f, out_f), b in zip(dims, biases)
        )
        self.lifs = nn.ModuleList(
            snn.Leaky(
                beta=beta,
                threshold=thr,
                reset_mechanism=rm,
                reset_delay=rd,
                graded_spikes_factor=gsf,
                inhibition=inhib,
                state_quant=sq,
            )
            for beta, thr, rm, rd, gsf, inhib, sq in zip(
                betas, thresholds, reset_mechanisms, reset_delays, graded_spikes_factors,
                inhibitions, state_quants,
            )
        )

    def forward(self, x):
        mems = [lif.reset_mem() for lif in self.lifs]
        value = x
        spk = None
        for i, (linear, lif) in enumerate(zip(self.linears, self.lifs)):
            cur = linear(value)
            spk, mems[i] = lif(cur, mems[i])
            value = spk
        return spk, mems[-1]


def _set_linear(linear: nn.Linear, weight, bias=None) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        if bias is not None:
            linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))


def _run_snntorch_reference(model: ChainSNN, inputs):
    """Drives an already-independent snnTorch model instance layer-by-layer
    (not through model.forward, so we can capture every layer's per-step
    current/spike/mem, not just the final output) and returns per-timestep
    (currents, spikes, mems) lists in IR layer order."""
    for lif in model.lifs:
        lif.reset_mem()

    all_currents, all_spikes, all_mems = [], [], []
    for x in inputs:
        value = x
        currents, spikes, mems = [], [], []
        for linear, lif in zip(model.linears, model.lifs):
            cur = linear(value)
            currents.append(cur)
            spk, mem = lif(cur, lif.mem)
            spikes.append(spk)
            mems.append(mem)
            value = spk
        all_currents.append(currents)
        all_spikes.append(spikes)
        all_mems.append(mems)
    return all_currents, all_spikes, all_mems


def _assert_equivalent(ir, inputs, model_for_reference: ChainSNN):
    """Parses `ir`'s originating model is assumed already done by the
    caller; this runs both executors over `inputs` and compares every
    per-layer current/spike/mem value at every timestep."""
    ir_outputs, ir_final_state, ir_traces = run_sequence(ir, inputs)
    snn_currents, snn_spikes, snn_mems = _run_snntorch_reference(model_for_reference, inputs)

    assert len(ir_traces) == len(inputs)
    for t, (ir_trace, cur_t, spk_t, mem_t) in enumerate(
        zip(ir_traces, snn_currents, snn_spikes, snn_mems)
    ):
        for i, (a, b) in enumerate(zip(ir_trace.currents, cur_t)):
            assert torch.allclose(a, b, atol=ATOL), f"current mismatch t={t} layer={i}"
        for i, (a, b) in enumerate(zip(ir_trace.spikes, spk_t)):
            assert torch.equal(a, b), f"spike mismatch t={t} layer={i}: {a} vs {b}"
        for i, (a, b) in enumerate(zip(ir_trace.mem, mem_t)):
            assert torch.allclose(a, b, atol=ATOL), f"mem mismatch t={t} layer={i}"

    # final IR output must equal the last LIF layer's spike this timestep
    assert torch.equal(ir_outputs[-1], snn_spikes[-1][-1])
    assert torch.allclose(ir_final_state[-1], snn_mems[-1][-1], atol=ATOL)


def test_tiny_linear_leaky_manual_weights_crossing_and_decay():
    """A single Linear(2,2) -> Leaky with hand-chosen weights, driven
    through excitation (crossing threshold, resetting -- possibly more
    than once) and then zero input (pure membrane decay)."""
    model = ChainSNN(dims=[(2, 2)], betas=[0.9], thresholds=[1.0])
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)

    inputs = [
        torch.tensor([[1.2, 0.3]]),  # channel 0 crosses threshold immediately
        torch.tensor([[1.2, 0.3]]),  # crosses again after partial reset
        torch.tensor([[1.2, 0.3]]),  # and again
        torch.tensor([[0.0, 0.0]]),  # zero input: pure decay
        torch.tensor([[0.0, 0.0]]),  # more decay
        torch.tensor([[0.0, 0.0]]),
    ]

    _assert_equivalent(ir, inputs, model_ref)


def test_two_layer_linear_leaky_linear_leaky():
    """Linear(2,3) -> Leaky -> Linear(3,2) -> Leaky, deterministic (seeded)
    weights, mixed excitation/decay sequence, with bias=False on the
    second Linear layer."""
    torch.manual_seed(42)
    model = ChainSNN(
        dims=[(2, 3), (3, 2)],
        betas=[0.95, 0.85],
        thresholds=[1.0, 0.7],
        biases=[True, False],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[2].bias is None  # fc2 (bias=False) preserved through parsing

    torch.manual_seed(7)
    inputs = [torch.rand(3, 2) * 1.5 for _ in range(5)]
    inputs += [torch.zeros(3, 2) for _ in range(3)]  # trailing zero-input decay

    _assert_equivalent(ir, inputs, model_ref)


def test_deeper_three_layer_chain():
    """A deeper [Linear -> Leaky]^3 model, deterministic (seeded) weights
    and inputs, spanning many timesteps with excitation and decay."""
    torch.manual_seed(123)
    model = ChainSNN(
        dims=[(4, 6), (6, 4), (4, 2)],
        betas=[0.95, 0.90, 0.92],
        thresholds=[1.0, 0.8, 1.1],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)

    torch.manual_seed(99)
    inputs = [torch.rand(2, 4) * 2.0 for _ in range(6)]  # strong: forces crossings
    inputs += [torch.zeros(2, 4) for _ in range(4)]  # trailing decay

    _assert_equivalent(ir, inputs, model_ref)


def test_initialize_state_matches_zero_membrane_start():
    model = ChainSNN(dims=[(3, 5), (5, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)

    state = initialize_state(ir, batch_size=4)

    assert len(state) == 2
    assert state[0].shape == (4, 5)
    assert state[1].shape == (4, 2)
    assert torch.equal(state[0], torch.zeros(4, 5))
    assert torch.equal(state[1], torch.zeros(4, 2))


def test_step_executes_exactly_one_timestep_and_updates_state():
    model = ChainSNN(dims=[(2, 2)], betas=[0.9], thresholds=[1.0])
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    ir = parse_snntorch_model(model)

    state0 = initialize_state(ir, batch_size=1)
    x = torch.tensor([[1.2, 0.3]])

    out1, state1, trace1 = step(ir, x, state0)
    # beta*0 + 1.2 = 1.2 > 1.0 -> spike, mem = 1.2 - 1.0 = 0.2
    assert torch.equal(out1, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(state1[0], torch.tensor([[0.2, 0.3]]), atol=ATOL)

    out2, state2, trace2 = step(ir, x, state1)
    # beta*0.2 + 1.2 = 1.38 > 1.0 -> spike, mem = 1.38 - 1.0 = 0.38
    assert torch.equal(out2, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(state2[0], torch.tensor([[0.38, 0.57]]), atol=ATOL)

    # state0 must be untouched by step() (no in-place mutation of caller's state)
    assert torch.equal(state0[0], torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# reset_mechanism x reset_delay conformance. Each combination is compared against a live snnTorch 1.0.0
# Leaky instance, not just the previously-accepted subtract+immediate case.
# ---------------------------------------------------------------------------

_RESET_COMBINATIONS = [
    ("subtract", False),
    ("subtract", True),
    ("zero", False),
    ("zero", True),
    ("none", False),  # reset_delay is numerically irrelevant for "none"
]


def _reset_conformance_inputs():
    """Deterministic per-timestep current sequence, threshold=1.0 assumed,
    engineered to exercise: below threshold, near/at threshold, slightly
    above, a large supra-threshold jump, repeated firing, decay to zero
    input, and a channel driven negative throughout."""
    return [
        torch.tensor([[0.3, -0.3]]),   # both below threshold
        torch.tensor([[0.9, -0.5]]),   # channel 0 close to, still below, threshold
        torch.tensor([[1.05, -0.3]]),  # channel 0 slightly above
        torch.tensor([[5.0, -0.3]]),   # channel 0 large supra-threshold jump
        torch.tensor([[1.5, -0.3]]),   # channel 0 fires again (repeated firing)
        torch.tensor([[1.5, -0.3]]),   # and again
        torch.tensor([[0.0, -0.3]]),   # zero input -- decay begins
        torch.tensor([[0.0, -0.3]]),
        torch.tensor([[0.0, -0.3]]),
        torch.tensor([[0.0, -2.0]]),   # push channel 1 further negative
    ]


@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_reset_conformance_single_layer(reset_mechanism, reset_delay):
    """For each of the 5 distinct (reset_mechanism, reset_delay) numerical
    behaviors, compares the IR reference executor against live
    snntorch.Leaky over a rich stimulus sequence (below/near/above
    threshold, a large supra-threshold jump, repeated firing, decay,
    negative membrane) -- not merely a final-prediction check."""
    model = ChainSNN(
        dims=[(2, 2)],
        betas=[0.9],
        thresholds=[1.0],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
    )
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].reset_mechanism == reset_mechanism
    assert ir.layers[1].reset_delay is reset_delay

    _assert_equivalent(ir, _reset_conformance_inputs(), model_ref)


def test_strict_greater_than_threshold_does_not_fire_at_exact_equality():
    """snnTorch's fire()/mem_reset() both use a strict `mem - threshold > 0`
    comparison (the ATan surrogate's forward pass is `(input_ > 0).float()`) -- mem exactly equal to threshold must
    NOT fire, for every reset_mechanism. Hand-constructed single step:
    beta=1.0, mem=0 initially, current=1.0 -> pre-spike mem lands exactly
    at threshold=1.0."""
    for reset_mechanism in ("subtract", "zero", "none"):
        for reset_delay in (False, True):
            model = ChainSNN(
                dims=[(1, 1)],
                betas=[1.0],
                thresholds=[1.0],
                reset_mechanisms=[reset_mechanism],
                reset_delays=[reset_delay],
            )
            _set_linear(model.linears[0], weight=[[1.0]], bias=[0.0])
            model_ref = copy.deepcopy(model)
            ir = parse_snntorch_model(model)

            inputs = [torch.tensor([[1.0]])]  # mem lands exactly at threshold
            _assert_equivalent(ir, inputs, model_ref)

            # explicit: neither executor fired at exact equality
            _, _, traces = run_sequence(ir, inputs)
            assert torch.equal(traces[0].spikes[0], torch.tensor([[0.0]]))


# ---------------------------------------------------------------------------
# scalar/per-neuron beta+threshold conformance, crossed with every reset combination.
# ---------------------------------------------------------------------------


def _per_neuron_conformance_inputs():
    """Deterministic per-timestep current sequence for a 4-neuron layer,
    engineered (with beta=[0.0,0.4,0.8,1.0], threshold=[0.3,0.6,0.9,1.2])
    so each neuron behaves visibly differently: neuron 0 (beta=0, an
    instant integrator with no memory) fires/resets every step it gets
    input; neuron 3 (beta=1.0, a pure accumulator) only ever climbs;
    neurons 1-2 decay at different rates. Exercises negative membrane,
    exact-threshold, supra-threshold, zero-input decay, and repeated
    firing across the 4 neurons collectively."""
    return [
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]),   # all four get identical input, diverge by beta
        torch.tensor([[0.3, 0.3, 0.3, 0.3]]),   # neuron0 exactly at its own threshold (0.3)
        torch.tensor([[-0.2, -0.2, -0.2, -0.2]]),  # negative input -- pushes membranes down
        torch.tensor([[2.0, 2.0, 2.0, 2.0]]),   # supra-threshold for all
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]),   # repeated firing region
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),   # zero input -- decay
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, -1.0, -1.0, -1.0]]),  # drive negative again
    ]


@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_per_neuron_beta_threshold_reset_conformance(reset_mechanism, reset_delay):
    """Per-neuron beta/threshold crossed with all 5 reset combinations, against live snnTorch.Leaky."""
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
    )
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].beta == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert ir.layers[1].threshold == pytest.approx([0.3, 0.6, 0.9, 1.2])

    _assert_equivalent(ir, _per_neuron_conformance_inputs(), model_ref)


# ---------------------------------------------------------------------------
# graded_spikes_factor conformance.
# Reuses _RESET_COMBINATIONS and _per_neuron_conformance_inputs from the
# sections above, so graded spikes are checked with every reset mode and per-neuron parameters.
# ---------------------------------------------------------------------------


def _graded_spikes_conformance_inputs():
    """Deterministic per-timestep current sequence for a 4-neuron layer
    with graded_spikes_factor=[0.25, 0.75, 1.5, 2.5] and beta=1.0 (pure
    accumulator, threshold=1.0 for every neuron): engineered so some
    neurons never fire, some fire once, some fire repeatedly, and
    different neurons fire on different timesteps."""
    return [
        torch.tensor([[0.4, 0.4, 0.4, 0.4]]),   # nobody crosses threshold=1.0 yet
        torch.tensor([[0.7, 0.7, 0.7, 0.7]]),   # mem=1.1 for all -- everyone fires, resets (subtract)
        torch.tensor([[0.2, 0.2, 0.2, 0.2]]),   # below threshold again after reset
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),   # never fires again -- some neurons stay silent
        torch.tensor([[1.5, 1.5, 1.5, 1.5]]),   # large jump -- fires again
        torch.tensor([[1.5, 1.5, 1.5, 1.5]]),   # fires again -- repeated firing
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
    ]


def test_graded_spikes_conformance_diagnostic_network():
    """A tiny 4-neuron diagnostic network with distinct
    graded_spikes_factor per neuron, compared timestep-by-timestep against
    live snnTorch for current/spike(emitted)/membrane, AND an explicit
    check that every emitted spike value is exactly 0 or that neuron's own
    factor (never the raw binary 0/1)."""
    factors = [0.25, 0.75, 1.5, 2.5]
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[1.0],
        thresholds=[1.0],
        graded_spikes_factors=[factors],
    )
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].graded_spikes_factor == pytest.approx(factors)

    inputs = _graded_spikes_conformance_inputs()
    _assert_equivalent(ir, inputs, model_ref)

    _, _, traces = run_sequence(ir, inputs)
    factors_t = torch.tensor(factors)
    for t, trace in enumerate(traces):
        spk = trace.spikes[0][0]  # batch size 1
        for j in range(4):
            assert spk[j].item() in (0.0, pytest.approx(factors_t[j].item())), (
                f"t={t} neuron={j}: emitted spike {spk[j].item()} is neither 0 "
                f"nor factor[{j}]={factors_t[j].item()}"
            )


@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_graded_spikes_reset_conformance(reset_mechanism, reset_delay):
    """A scalar graded_spikes_factor (!= 1.0) crossed with all
    5 reset behaviors -- proves that scaling the emitted spike does NOT
    change reset magnitude/order (compared against live snnTorch, not just
    re-asserting the formula)."""
    model = ChainSNN(
        dims=[(2, 2)],
        betas=[0.9],
        thresholds=[1.0],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
        graded_spikes_factors=[2.5],
    )
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].graded_spikes_factor == pytest.approx([2.5])

    _assert_equivalent(ir, _reset_conformance_inputs(), model_ref)


def test_graded_spikes_per_neuron_beta_threshold_composition():
    """Per-neuron beta + per-neuron threshold + per-neuron
    graded_spikes_factor combined in one model, compared against live
    snnTorch -- proves reset modes, per-neuron parameters and graded spikes compose correctly rather than only
    being tested in isolation."""
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5]],
    )
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].beta == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert ir.layers[1].threshold == pytest.approx([0.3, 0.6, 0.9, 1.2])
    assert ir.layers[1].graded_spikes_factor == pytest.approx([0.25, 0.75, 1.5, 2.5])

    _assert_equivalent(ir, _per_neuron_conformance_inputs(), model_ref)


# ---------------------------------------------------------------------------
# inhibition (layer-wide winner-take-all) conformance, crossed with every reset combination.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_inhibition_reset_conformance_two_neurons_cross_simultaneously(reset_mechanism, reset_delay):
    """At least two neurons simultaneously cross
    threshold, compared against live snnTorch for all 5 reset
    combinations -- specifically checks the inhibition/immediate-reset
    interaction in snnTorch's source (membrane
    DOES differ from non-inhibited once reset_delay=False)."""
    model = ChainSNN(
        dims=[(3, 3)],
        betas=[1.0],
        thresholds=[1.0],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
        inhibitions=[True],
    )
    _set_linear(model.linears[0], weight=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias=[0.0, 0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].inhibition is True

    inputs = [
        torch.tensor([[1.5, 1.3, 0.2]]),   # two neurons already above threshold incoming
        torch.tensor([[0.6, 0.6, 0.6]]),   # both neuron0/neuron1 cross again -- only one may win
        torch.tensor([[0.6, 0.6, 0.6]]),   # repeated
        torch.tensor([[0.0, 0.0, 0.0]]),   # decay
        torch.tensor([[2.0, 1.9, -0.5]]),  # large supra-threshold jump, neuron2 pushed negative
    ]
    _assert_equivalent(ir, inputs, model_ref)


def test_inhibition_winner_selected_by_margin_not_raw_membrane():
    """Winner selection must use (mem - threshold), not
    raw membrane -- a per-neuron threshold configuration where the highest
    RAW membrane neuron is NOT the highest-MARGIN neuron. A mistaken
    raw-membrane argmax implementation would pick the wrong winner and
    fail this equivalence check against live snnTorch."""
    model = ChainSNN(
        dims=[(3, 3)],
        betas=[1.0],
        thresholds=[[0.1, 5.0, 0.1]],  # neuron1 has a huge threshold
        inhibitions=[True],
    )
    _set_linear(model.linears[0], weight=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias=[0.0, 0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)

    # raw membrane: neuron1 highest (4.0); margin: n0=0.4, n1=-1.0, n2=-0.05
    # -- highest MARGIN is neuron0, not neuron1 (highest raw membrane)
    inputs = [torch.tensor([[0.5, 4.0, 0.05]])]
    _assert_equivalent(ir, inputs, model_ref)

    _, _, traces = run_sequence(ir, inputs)
    assert torch.equal(traces[0].spikes[0], torch.tensor([[1.0, 0.0, 0.0]]))


def test_inhibition_exact_margin_tie_matches_snntorch():
    """An exact tie in margin across 4 neurons --
    whichever index live snnTorch actually picks, the IR must pick the
    same one (documented as matching the tested torch version's observed
    tie-breaking, not a broader guarantee)."""
    model = ChainSNN(dims=[(4, 4)], betas=[1.0], thresholds=[1.0], inhibitions=[True])
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)
    ir = parse_snntorch_model(model)

    inputs = [torch.tensor([[1.5, 1.5, 1.5, 1.5]])]  # all four exactly tied
    _assert_equivalent(ir, inputs, model_ref)


def test_inhibition_graded_spikes_factor_composition():
    """Inhibition combined with a per-neuron
    graded_spikes_factor != 1 -- snnTorch's
    `fire_inhibition()` never applies graded_spikes_factor to its output
    (unlike `fire()`), so the emitted spike must be exactly the winner-
    masked BINARY decision, not scaled -- and the immediate-reset
    interaction with graded_spikes_factor (which does NOT algebraically
    cancel under inhibition, unlike the non-inhibited case) must match
    live snnTorch exactly, not an assumed formula."""
    model = ChainSNN(
        dims=[(3, 3)],
        betas=[1.0],
        thresholds=[1.0],
        reset_mechanisms=["subtract"],
        reset_delays=[False],
        graded_spikes_factors=[[0.25, 1.0, 2.0]],
        inhibitions=[True],
    )
    _set_linear(model.linears[0], weight=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias=[0.0, 0.0, 0.0])
    model_ref = copy.deepcopy(model)
    ir = parse_snntorch_model(model)

    inputs = [
        torch.tensor([[1.5, 1.3, 0.0]]),  # neuron0 highest margin, factor=0.25
        torch.tensor([[0.6, 0.6, 0.6]]),
    ]
    _assert_equivalent(ir, inputs, model_ref)

    _, _, traces = run_sequence(ir, inputs)
    # emitted spikes must be exactly 0 or 1 (never scaled by graded factor)
    for trace in traces:
        for v in trace.spikes[0].flatten().tolist():
            assert v in (0.0, 1.0), f"inhibited emitted spike {v} is not binary"


def test_inhibition_batching_selects_winner_per_row():
    """Winner selection must be independent per batch
    row, not global across the batch."""
    model = ChainSNN(dims=[(3, 3)], betas=[1.0], thresholds=[1.0], inhibitions=[True])
    _set_linear(model.linears[0], weight=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias=[0.0, 0.0, 0.0])
    model_ref = copy.deepcopy(model)
    ir = parse_snntorch_model(model)

    inputs = [torch.tensor([[1.5, 0.2, 0.1], [0.1, 0.2, 1.5]])]
    _assert_equivalent(ir, inputs, model_ref)

    _, _, traces = run_sequence(ir, inputs)
    assert torch.equal(traces[0].spikes[0], torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def _inhibition_diagnostic_inputs():
    """4-neuron sequence engineered to
    exercise: no neurons firing, exactly one firing, several firing,
    winner changes across timesteps, repeated firing, and old_mem already
    above threshold at the start of a step."""
    return [
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),    # no neurons firing
        torch.tensor([[0.35, 0.0, 0.0, 0.0]]),   # exactly one firing (neuron0, threshold=0.3)
        torch.tensor([[0.05, 0.65, 0.0, 0.0]]),  # winner shifts to neuron1
        torch.tensor([[0.3, 0.3, 1.0, 0.0]]),    # several neurons' margins positive at once
        torch.tensor([[0.3, 0.3, 0.3, 1.3]]),    # neuron3 (highest threshold) now wins
        torch.tensor([[0.3, 0.3, 0.3, 1.3]]),    # repeated firing region
        torch.tensor([[0.9, 0.9, 0.9, 0.9]]),    # old_mem already above threshold going into this step
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),    # decay
    ]


@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_inhibition_diagnostic_conformance(reset_mechanism, reset_delay):
    """A rich 4-neuron diagnostic sequence crossed
    with all 5 reset combinations, per-neuron beta/threshold, compared
    against live snnTorch timestep-by-timestep (not merely final
    prediction)."""
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
        inhibitions=[True],
    )
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    _assert_equivalent(ir, _inhibition_diagnostic_inputs(), model_ref)


# ---------------------------------------------------------------------------
# state_quant conformance. Reuses
# _RESET_COMBINATIONS/_inhibition_diagnostic_inputs from the sections above.
# ---------------------------------------------------------------------------


def _closure_levels(q):
    return q.__closure__[q.__code__.co_freevars.index("levels")].cell_contents


# Item A: basic quantizer behavior (direct unit test of _quantize_state,
# independent of any Leaky model) -- exact level stays exact, nearest
# selection between levels, exact midpoint tie, below-lowest/above-highest
# clamping.
def test_quantize_state_basic_behavior():
    levels = (-1.0, -1 / 3, 1 / 3, 1.0)

    def q(x):
        return _quantize_state(torch.tensor([[x]]), levels).item()

    assert q(-1.0) == pytest.approx(-1.0)          # exact level stays exact
    assert q(1 / 3) == pytest.approx(1 / 3)         # exact level stays exact
    assert q(0.5) == pytest.approx(1 / 3)           # nearest: closer to 1/3 than 1.0
    assert q(-0.5) == pytest.approx(-1 / 3)         # nearest: closer to -1/3 than -1.0
    assert q(0.0) == pytest.approx(-1 / 3)          # exact midpoint tie -> lower/first level wins
    assert q(-5.0) == pytest.approx(-1.0)           # below lowest -> clamps to lowest level
    assert q(5.0) == pytest.approx(1.0)             # above highest -> clamps to highest level


# Item F: uniform quantization at representative bit widths.
@pytest.mark.parametrize("num_bits", [2, 4, 8])
def test_uniform_state_quant_conformance(num_bits):
    """Uniform quantizer at 2/4/8-bit widths, compared
    against live snnTorch over a stimulus sequence engineered to cross
    threshold both before and after quantization changes the outcome."""
    q = quant.state_quant(num_bits=num_bits, uniform=True, threshold=1.0, lower_limit=0, upper_limit=0.5)
    model = ChainSNN(
        dims=[(2, 2)], betas=[1.0], thresholds=[1.0],
        reset_mechanisms=["subtract"], reset_delays=[False], state_quants=[q],
    )
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].state_quant is not None
    assert len(ir.layers[1].state_quant.levels) == 2 ** num_bits

    inputs = [
        torch.tensor([[0.2, -0.2]]),
        torch.tensor([[0.6, 0.1]]),
        torch.tensor([[1.3, 0.9]]),
        torch.tensor([[1.3, 0.9]]),
        torch.tensor([[0.0, 0.0]]),
    ]
    _assert_equivalent(ir, inputs, model_ref)


# Item G: non-uniform quantization -- both thr_centered modes, plus a
# non-default multiplier.
@pytest.mark.parametrize(
    "quant_kwargs",
    [
        dict(num_bits=3, uniform=False, thr_centered=True, threshold=1.0),
        dict(num_bits=3, uniform=False, thr_centered=False, threshold=1.0),
        dict(num_bits=4, uniform=False, thr_centered=True, threshold=1.0, multiplier=0.6),
    ],
    ids=["thr_centered_true", "thr_centered_false", "custom_multiplier"],
)
def test_nonuniform_state_quant_conformance(quant_kwargs):
    q = quant.state_quant(**quant_kwargs)
    model = ChainSNN(
        dims=[(2, 2)], betas=[1.0], thresholds=[1.0],
        reset_mechanisms=["zero"], reset_delays=[False], state_quants=[q],
    )
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    inputs = [
        torch.tensor([[0.85, 0.3]]),
        torch.tensor([[1.2, -0.5]]),
        torch.tensor([[0.0, 0.0]]),
    ]
    _assert_equivalent(ir, inputs, model_ref)


# Item C: reset x state_quant matrix, single neuron.
@pytest.mark.parametrize("reset_mechanism,reset_delay", _RESET_COMBINATIONS)
def test_state_quant_reset_conformance(reset_mechanism, reset_delay):
    """All 5 reset combinations, quantized, vs live snnTorch
    -- specifically checks whether the immediate-reset final membrane is
    or isn't itself re-quantized (it must NOT be, per the snnTorch source
    re-audit)."""
    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0, lower_limit=0, upper_limit=0.5)
    model = ChainSNN(
        dims=[(1, 1)], betas=[1.0], thresholds=[1.0],
        reset_mechanisms=[reset_mechanism], reset_delays=[reset_delay], state_quants=[q],
    )
    _set_linear(model.linears[0], weight=[[1.0]], bias=[0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    inputs = [torch.tensor([[1.3]]), torch.tensor([[1.3]]), torch.tensor([[0.0]])]
    _assert_equivalent(ir, inputs, model_ref)


# Item D: inhibition x state_quant -- winner selection must follow the
# QUANTIZED margin, with a case where the raw-membrane winner would differ
# from the quantized-membrane winner, using per-neuron thresholds.
def test_inhibition_state_quant_winner_follows_quantized_margin():
    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0, lower_limit=0, upper_limit=0.5)
    model = ChainSNN(
        dims=[(3, 3)], betas=[1.0], thresholds=[[1.0, 1.0, 0.5]],
        reset_mechanisms=["subtract"], reset_delays=[False],
        inhibitions=[True], state_quants=[q],
    )
    _set_linear(model.linears[0], weight=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias=[0.0, 0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].inhibition is True
    assert ir.layers[1].state_quant is not None

    inputs = [
        torch.tensor([[0.95, 0.65, 0.55]]),
        torch.tensor([[0.3, 1.4, 0.2]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
    ]
    _assert_equivalent(ir, inputs, model_ref)


# graded_spikes_factor x state_quant, for both inhibition=False and
# inhibition=True -- must preserve the exact behavior (spk/factor cancellation for non-inhibited layers; the
# non-cancelling literal division for inhibited layers).
@pytest.mark.parametrize("inhibition", [False, True])
def test_graded_state_quant_composition(inhibition):
    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0, lower_limit=0, upper_limit=0.5)
    model = ChainSNN(
        dims=[(2, 2)], betas=[1.0], thresholds=[1.0],
        reset_mechanisms=["subtract"], reset_delays=[False],
        graded_spikes_factors=[2.5], inhibitions=[inhibition], state_quants=[q],
    )
    _set_linear(model.linears[0], weight=[[1.0, 0.0], [0.0, 1.0]], bias=[0.0, 0.0])
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    inputs = [torch.tensor([[1.3, 0.2]]), torch.tensor([[0.0, 0.0]])]
    _assert_equivalent(ir, inputs, model_ref)


# Item I: full-composition torture test -- per-neuron beta/threshold/
# graded_spikes_factor, inhibition=True, state_quant enabled, reset_delay
# =False, all in one model, compared timestep-by-timestep against live
# snnTorch.
def test_full_leaky_feature_composition():
    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0, lower_limit=0, upper_limit=0.5)
    model = ChainSNN(
        dims=[(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5]],
        reset_mechanisms=["subtract"], reset_delays=[False],
        inhibitions=[True], state_quants=[q],
    )
    _set_linear(
        model.linears[0],
        weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bias=[0.0, 0.0, 0.0, 0.0],
    )
    model_ref = copy.deepcopy(model)

    ir = parse_snntorch_model(model)
    assert ir.layers[1].beta == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert ir.layers[1].threshold == pytest.approx([0.3, 0.6, 0.9, 1.2])
    assert ir.layers[1].graded_spikes_factor == pytest.approx([0.25, 0.75, 1.5, 2.5])
    assert ir.layers[1].inhibition is True
    assert ir.layers[1].state_quant is not None

    _assert_equivalent(ir, _inhibition_diagnostic_inputs(), model_ref)
