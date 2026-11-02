"""Legality checks for the source IR.

The frontend rejects what it cannot extract from a live ``nn.Module``; this verifier checks an
already built ``SNNModel`` value, however it was produced (frontend, hand-built, deserialized).
It is target independent: hardware limits are a backend concern.
"""


import math


from typing import List


import torch


from spikeduet.ir.model import AdaptiveThresholdSpec, LIFLayer, LinearLayer, SNNModel, StateQuantSpec


_SUPPORTED_RESET_MECHANISMS = ("subtract", "zero", "none")


class IRVerificationError(Exception):
    """An ``SNNModel`` violates a source-IR invariant. The message names the offending layer."""


def _fail(detail: str) -> None:
    raise IRVerificationError(f"IR verification failed:\n{detail}")


def _check_finite(label: str, values) -> None:
    for v in values:
        if not math.isfinite(v):
            _fail(f"{label} contains a non-finite value ({v!r})")


def _verify_linear(index: int, layer: LinearLayer) -> None:
    if layer.in_features <= 0 or layer.out_features <= 0:
        _fail(
            f"Linear layer {index} has non-positive dimensions "
            f"(in_features={layer.in_features}, out_features={layer.out_features})"
        )

    if layer.weight.dim() != 2:
        _fail(
            f"Linear layer {index} weight has rank {layer.weight.dim()} "
            "(expected rank 2, shape [out_features, in_features])"
        )
    expected_shape = (layer.out_features, layer.in_features)
    if tuple(layer.weight.shape) != expected_shape:
        _fail(
            f"Linear layer {index} weight has shape {tuple(layer.weight.shape)}\n"
            f"but expected {list(expected_shape)} ([out_features, in_features])"
        )
    _check_finite(f"Linear layer {index} weight", layer.weight.reshape(-1).tolist())

    if layer.bias is not None:
        expected_bias_shape = (layer.out_features,)
        if tuple(layer.bias.shape) != expected_bias_shape:
            _fail(
                f"Linear layer {index} bias has shape {tuple(layer.bias.shape)}\n"
                f"but expected {list(expected_bias_shape)}"
            )
        _check_finite(f"Linear layer {index} bias", layer.bias.reshape(-1).tolist())

    if layer.synapse_delays is not None:
        d = layer.synapse_delays
        if not isinstance(d, torch.Tensor) or d.dtype.is_floating_point or d.dtype == torch.bool:
            _fail(f"Linear layer {index} synapse_delays must be an integer tensor")
        if tuple(d.shape) != expected_shape:
            _fail(
                f"Linear layer {index} synapse_delays has shape {tuple(d.shape)}\n"
                f"but expected {list(expected_shape)} ([out_features, in_features])"
            )
        if int(d.min()) < 0:
            _fail(f"Linear layer {index} synapse_delays contains a negative delay ({int(d.min())})")
        if int(d.max()) > 255:
            _fail(f"Linear layer {index} synapse_delays exceeds the supported maximum delay of 255 ({int(d.max())})")


def _verify_param(index: int, param_name: str, values: List[float], num_neurons: int) -> None:
    if len(values) not in (1, num_neurons):
        _fail(f"LIF layer {index} {param_name} has length {len(values)}\nbut neuron count is {num_neurons}")
    _check_finite(f"LIF layer {index} {param_name}", values)


def _verify_state_quant(index: int, spec: StateQuantSpec) -> None:
    if not isinstance(spec, StateQuantSpec):
        _fail(f"LIF layer {index} state_quant is not a StateQuantSpec ({spec!r})")
    if len(spec.levels) == 0:
        _fail(f"LIF layer {index} state_quant has an empty level table")
    _check_finite(f"LIF layer {index} state_quant levels", spec.levels)
    # Duplicate levels are harmless; a descending pair would flip the lower-level-wins tie-break.
    levels = spec.levels
    if any(levels[i] > levels[i + 1] for i in range(len(levels) - 1)):
        _fail(
            f"LIF layer {index} state_quant levels are not in ascending order "
            f"{list(levels)!r} -- required for the documented nearest-level "
            "tie-break (the first/lowest-index level wins an exact tie)"
        )


def _verify_lif(index: int, layer: LIFLayer, num_neurons: int) -> None:
    _verify_param(index, "beta", layer.beta, num_neurons)
    _verify_param(index, "threshold", layer.threshold, num_neurons)
    _verify_param(index, "graded_spikes_factor", layer.graded_spikes_factor, num_neurons)

    if layer.reset_mechanism not in _SUPPORTED_RESET_MECHANISMS:
        _fail(
            f"LIF layer {index} has unsupported reset_mechanism "
            f"{layer.reset_mechanism!r}\n"
            f"expected one of {list(_SUPPORTED_RESET_MECHANISMS)!r}"
        )
    if not isinstance(layer.reset_delay, bool):
        _fail(f"LIF layer {index} reset_delay is not a bool ({layer.reset_delay!r})")
    if not isinstance(layer.inhibition, bool):
        _fail(f"LIF layer {index} inhibition is not a bool ({layer.inhibition!r})")

    if layer.state_quant is not None:
        _verify_state_quant(index, layer.state_quant)

    if not isinstance(layer.spike_state, bool):
        _fail(f"LIF layer {index} spike_state is not a bool ({layer.spike_state!r})")
    if layer.adaptation is not None:
        _verify_adaptation(index, layer, num_neurons)
    if layer.spike_state:
        _verify_spike_state_combo(index, layer, "spike_state")


def _verify_adaptation(index: int, layer: LIFLayer, num_neurons: int) -> None:
    spec = layer.adaptation
    if not isinstance(spec, AdaptiveThresholdSpec):
        _fail(f"LIF layer {index} adaptation is not an AdaptiveThresholdSpec ({spec!r})")
    _verify_param(index, "adaptation.rho", list(spec.rho), num_neurons)
    _verify_param(index, "adaptation.a", list(spec.a), num_neurons)
    if any(r < 0.0 or r > 1.0 for r in spec.rho):
        _fail(f"LIF layer {index} adaptation.rho must lie in [0, 1], got {list(spec.rho)!r}")
    if any(a < 0.0 for a in spec.a):
        _fail(f"LIF layer {index} adaptation.a must be >= 0 (resolve |a| at export), got {list(spec.a)!r}")
    if not layer.spike_state:
        _fail(f"LIF layer {index} uses adaptation (ALIF) but spike_state is False; adaptation is defined on the stored previous spike")
    _verify_spike_state_combo(index, layer, "adaptation (ALIF)")


def _verify_spike_state_combo(index: int, layer: LIFLayer, what: str) -> None:
    # The ALIF recurrence (see AdaptiveThresholdSpec) is derived ONLY for
    # this combination; every other one would need its own derivation.
    unsupported = []
    if layer.reset_mechanism != "subtract":
        unsupported.append(f"reset_mechanism={layer.reset_mechanism!r}")
    if not layer.reset_delay:
        unsupported.append("reset_delay=False")
    if layer.inhibition:
        unsupported.append("inhibition=True")
    if layer.state_quant is not None:
        unsupported.append("state_quant")
    if any(g != 1.0 for g in layer.graded_spikes_factor):
        unsupported.append("graded_spikes_factor != 1")
    if unsupported:
        _fail(
            f"LIF layer {index} combines {what} with {', '.join(unsupported)}; "
            "this is only defined for reset_mechanism='subtract', "
            "reset_delay=True, no inhibition, no state_quant, unit graded_spikes_factor"
        )


def verify_module_ir(model: SNNModel) -> None:
    """Verifies that `model` is an internally valid, supported
    `[Linear -> Leaky]^N` source IR value.

    Checks (see this module's docstring for the frontend/verifier
    boundary):

    Topology:
      * at least one layer; the chain starts with a Linear layer and ends
        with a LIF layer
      * strict alternation, no other layer kinds
      * each Linear layer's `in_features` matches the previous Linear
        layer's `out_features` (a LIF layer is dimension-preserving, so
        this is equivalent to "Leaky output dimension matches the next
        Linear layer's input dimension")

    Linear:
      * positive dimensions
      * weight rank 2, shape `[out_features, in_features]`
      * bias, if present, shape `[out_features]`
      * all weight/bias values finite

    LIF:
      * beta/threshold/graded_spikes_factor each have length 1 (shared
        scalar) or exactly the layer's neuron count (the preceding
        Linear's `out_features`), and are finite
      * `reset_mechanism` in `{"subtract", "zero", "none"}`
      * `reset_delay`/`inhibition` are bools
      * `state_quant`, if set, is a `StateQuantSpec` with a non-empty,
        finite, ascending level table

    Does not check anything platform/target-specific (buffer sizes,
    alignment, accelerator support, numeric-type support, ...) -- see
    `compiler/target/platform.py::PlatformSpec` for that, a deliberately
    separate, later (B2+) concern: a `Linear(784, 64)` layer is valid
    source IR even for a target whose accelerator cannot execute that
    matmul directly.

    Raises:
        IRVerificationError: on the first invariant violated, with a
            diagnostic naming the offending layer.
    """
    if not model.layers:
        _fail("model has no layers")

    if not isinstance(model.layers[0], LinearLayer):
        _fail(
            "model must begin with a Linear layer, found "
            f"{type(model.layers[0]).__name__}"
        )
    if not isinstance(model.layers[-1], LIFLayer):
        _fail(
            "model must end with a LIF (Leaky) layer, found "
            f"{type(model.layers[-1]).__name__}"
        )

    expect_linear = True
    linear_index = 0
    lif_index = 0
    prev_linear: LinearLayer = None

    for pos, layer in enumerate(model.layers):
        if expect_linear:
            if not isinstance(layer, LinearLayer):
                _fail(
                    f"layer at position {pos} is {type(layer).__name__}, expected "
                    "a Linear layer ([Linear -> Leaky]^N ordering requires a "
                    "Linear layer here)"
                )
            _verify_linear(linear_index, layer)
            if prev_linear is not None and prev_linear.out_features != layer.in_features:
                _fail(
                    f"Linear layer {linear_index - 1} output dimension = "
                    f"{prev_linear.out_features}\n"
                    f"but Linear layer {linear_index} input dimension = "
                    f"{layer.in_features}"
                )
            prev_linear = layer
            linear_index += 1
            expect_linear = False
        else:
            if not isinstance(layer, LIFLayer):
                _fail(
                    f"layer at position {pos} is {type(layer).__name__}, expected "
                    "a LIF (Leaky) layer ([Linear -> Leaky]^N ordering requires a "
                    "Leaky layer here)"
                )
            _verify_lif(lif_index, layer, num_neurons=prev_linear.out_features)
            lif_index += 1
            expect_linear = True

    if not expect_linear:
        _fail(
            "model ends with an unpaired Linear layer; every Linear layer must "
            "be followed by a LIF layer to complete a [Linear -> Leaky] block"
        )

    if model.readout_weights is not None:
        if len(model.readout_weights) == 0:
            _fail("readout_weights is empty")
        _check_finite("readout_weights", model.readout_weights)
        if model.num_steps is not None and len(model.readout_weights) != model.num_steps:
            _fail(
                f"readout_weights has length {len(model.readout_weights)} "
                f"but model.num_steps = {model.num_steps}"
            )
