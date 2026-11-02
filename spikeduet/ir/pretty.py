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
    """Renders `model` as a concise, human-readable summary.

    One line per Linear layer (dimensions, bias presence) and an indented
    block per LIF layer covering every field `LIFLayer` carries. Does not
    require `model` to have passed `spikeduet.ir.verifier.verify_module_ir`
    -- this is a debugging aid, not a correctness check -- but assumes the
    same `[Linear -> Leaky]^N` shape the rest of Backend/IR V1 does; an
    unrecognized layer kind is rendered inline rather than raising.
    """
    lines = ["SNNModel"]
    linear_index = 0
    lif_index = 0
    prev_linear: Optional[LinearLayer] = None

    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            lines.append(
                f"  Linear[{linear_index}]: {layer.in_features} -> "
                f"{layer.out_features}, bias={layer.bias is not None}"
                + (f", synapse_delays(0..{int(layer.synapse_delays.max())})" if layer.synapse_delays is not None else "")
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
            if layer.spike_state:
                lines.append("      spike_state=True (reset keyed on the stored previous spike)")
            if layer.adaptation is not None:
                lines.append(f"      {_format_param('adaptation.rho', list(layer.adaptation.rho), num_neurons)}")
                lines.append(f"      {_format_param('adaptation.a', list(layer.adaptation.a), num_neurons)}")
            lif_index += 1
        else:
            lines.append(f"  <unrecognized layer: {type(layer).__name__}>")

    if model.num_steps is not None:
        lines.append(f"  num_steps={model.num_steps}")
    if model.readout is not None:
        lines.append(f"  readout={model.readout!r}")
    if model.readout_weights is not None:
        lines.append(f"  readout_weights=per_timestep(n={len(model.readout_weights)}, {_preview(model.readout_weights)})")

    return "\n".join(lines)


def module_ir_to_dict(model: SNNModel) -> dict:
    """Deterministic, JSON-serializable structural summary of `model`.

    Same information `format_module_ir` shows, as a plain dict of shapes
    and hyperparameters -- e.g. for generating the hand-written
    `validation/*/model_meta.json` metadata convention programmatically.
    Never includes weight/bias/beta/threshold/graded_spikes_factor/
    state_quant *values*, only their lengths -- this is a structural
    summary, not a weight-serialization format (`compiler/codegen.py`
    already owns the one place full numeric values are emitted).
    """
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
                    **({"synapse_delay_max": int(layer.synapse_delays.max())} if layer.synapse_delays is not None else {}),
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
                    **({"spike_state": True} if layer.spike_state else {}),
                    **({"adaptation_len": len(layer.adaptation.rho)} if layer.adaptation is not None else {}),
                }
            )
        else:
            layers.append({"kind": type(layer).__name__})

    return {
        "layers": layers,
        "num_steps": model.num_steps,
        "readout": model.readout,
        **({"readout_weights_len": len(model.readout_weights)} if model.readout_weights is not None else {}),
    }
