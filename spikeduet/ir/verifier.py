"""Legality checks for the source IR.

The frontend rejects what it cannot extract from a live ``nn.Module``; this verifier checks an
already built ``SNNModel`` value, however it was produced (frontend, hand-built, deserialized).
It is target independent: hardware limits are a backend concern.
"""

import math
from typing import List, Optional

from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel, StateQuantSpec

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
        if tuple(layer.bias.shape) != (layer.out_features,):
            _fail(
                f"Linear layer {index} bias has shape {tuple(layer.bias.shape)}\n"
                f"but expected {[layer.out_features]}"
            )
        _check_finite(f"Linear layer {index} bias", layer.bias.reshape(-1).tolist())


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
            f"LIF layer {index} has unsupported reset_mechanism {layer.reset_mechanism!r}\n"
            f"expected one of {list(_SUPPORTED_RESET_MECHANISMS)!r}"
        )
    if not isinstance(layer.reset_delay, bool):
        _fail(f"LIF layer {index} reset_delay is not a bool ({layer.reset_delay!r})")
    if not isinstance(layer.inhibition, bool):
        _fail(f"LIF layer {index} inhibition is not a bool ({layer.inhibition!r})")
    if layer.state_quant is not None:
        _verify_state_quant(index, layer.state_quant)


def verify_module_ir(model: SNNModel) -> None:
    """Checks that ``model`` is a valid ``[Linear -> LIF]^N`` chain.

    Topology: non-empty, strictly alternating Linear/LIF, starting with Linear and ending with
    LIF, and each Linear's ``in_features`` equal to the previous Linear's ``out_features``.
    Linear: positive dimensions, weight ``[out, in]``, bias ``[out]``, finite values.
    LIF: per-neuron parameters of length 1 or the neuron count, finite; a supported reset
    mechanism; boolean flags; a non-empty, finite, ascending ``state_quant`` table.

    Raises:
        IRVerificationError: on the first violation.
    """
    if not model.layers:
        _fail("model has no layers")
    if not isinstance(model.layers[0], LinearLayer):
        _fail(f"model must begin with a Linear layer, found {type(model.layers[0]).__name__}")
    if not isinstance(model.layers[-1], LIFLayer):
        _fail(f"model must end with a LIF (Leaky) layer, found {type(model.layers[-1]).__name__}")

    prev_linear: Optional[LinearLayer] = None
    for pos, layer in enumerate(model.layers):
        index = pos // 2
        if pos % 2 == 0:
            if not isinstance(layer, LinearLayer):
                _fail(
                    f"layer at position {pos} is {type(layer).__name__}, expected "
                    "a Linear layer ([Linear -> Leaky]^N ordering requires a "
                    "Linear layer here)"
                )
            _verify_linear(index, layer)
            if prev_linear is not None and prev_linear.out_features != layer.in_features:
                _fail(
                    f"Linear layer {index - 1} output dimension = {prev_linear.out_features}\n"
                    f"but Linear layer {index} input dimension = {layer.in_features}"
                )
            prev_linear = layer
        else:
            if not isinstance(layer, LIFLayer):
                _fail(
                    f"layer at position {pos} is {type(layer).__name__}, expected "
                    "a LIF (Leaky) layer ([Linear -> Leaky]^N ordering requires a "
                    "Leaky layer here)"
                )
            _verify_lif(index, layer, num_neurons=prev_linear.out_features)
