"""Human-readable and dict summaries of the source IR (shapes and hyperparameters, never weights)."""

from typing import List, Optional

from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel, StateQuantSpec

_PREVIEW_LIMIT = 4


def _preview(values) -> str:
    shown = ", ".join(f"{v:g}" for v in values[:_PREVIEW_LIMIT])
    if len(values) > _PREVIEW_LIMIT:
        shown += ", ..."
    return f"[{shown}]"


def _format_param(name: str, values: List[float], num_neurons: int) -> str:
    if len(values) == 1:
        return f"{name}=scalar({values[0]:g})"
    return f"{name}=per_neuron(n={num_neurons}, {_preview(values)})"


def _format_state_quant(spec: Optional[StateQuantSpec]) -> str:
    if spec is None:
        return "state_quant=None"
    return f"state_quant=levels(n={len(spec.levels)}, {_preview(spec.levels)})"


def format_module_ir(model: SNNModel) -> str:
    """One line per Linear layer and an indented block per LIF layer. Long per-neuron vectors are
    previewed. Does not verify the model."""
    lines = ["SNNModel"]
    linear_index = 0
    lif_index = 0
    prev_linear: Optional[LinearLayer] = None

    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            lines.append(
                f"  Linear[{linear_index}]: {layer.in_features} -> "
                f"{layer.out_features}, bias={layer.bias is not None}"
            )
            prev_linear = layer
            linear_index += 1
        elif isinstance(layer, LIFLayer):
            num_neurons = prev_linear.out_features if prev_linear is not None else -1
            lines.append(f"  Leaky[{lif_index}]: neurons={num_neurons}")
            lines.append(f"      {_format_param('beta', layer.beta, num_neurons)}")
            lines.append(f"      {_format_param('threshold', layer.threshold, num_neurons)}")
            lines.append(
                f"      {_format_param('graded_spikes_factor', layer.graded_spikes_factor, num_neurons)}"
            )
            lines.append(f"      reset={layer.reset_mechanism}")
            lines.append(f"      reset_delay={layer.reset_delay}")
            lines.append(f"      inhibition={layer.inhibition}")
            lines.append(f"      {_format_state_quant(layer.state_quant)}")
            lif_index += 1
        else:
            lines.append(f"  <unrecognized layer: {type(layer).__name__}>")

    if model.num_steps is not None:
        lines.append(f"  num_steps={model.num_steps}")
    if model.readout is not None:
        lines.append(f"  readout={model.readout!r}")

    return "\n".join(lines)


def module_ir_to_dict(model: SNNModel) -> dict:
    """JSON-serializable structural summary: dimensions and parameter lengths, never values."""
    layers = []
    prev_linear: Optional[LinearLayer] = None

    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            prev_linear = layer
            layers.append(
                {
                    "kind": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "bias": layer.bias is not None,
                }
            )
        elif isinstance(layer, LIFLayer):
            num_neurons = prev_linear.out_features if prev_linear is not None else None
            layers.append(
                {
                    "kind": "Leaky",
                    "neurons": num_neurons,
                    "beta_len": len(layer.beta),
                    "threshold_len": len(layer.threshold),
                    "graded_spikes_factor_len": len(layer.graded_spikes_factor),
                    "reset_mechanism": layer.reset_mechanism,
                    "reset_delay": layer.reset_delay,
                    "inhibition": layer.inhibition,
                    "state_quant_levels": (
                        len(layer.state_quant.levels) if layer.state_quant is not None else None
                    ),
                }
            )
        else:
            layers.append({"kind": type(layer).__name__})

    return {
        "layers": layers,
        "num_steps": model.num_steps,
        "readout": model.readout,
    }
