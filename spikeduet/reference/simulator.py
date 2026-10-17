"""Reference executor for the source IR.

Runs an ``SNNModel`` from the weights and parameters in the IR alone, without calling
``snntorch``. It is the numerical ground truth the generated C is compared against, and it is
itself tested against live ``snntorch.Leaky`` (``snntorch==1.0.0``).

Per neuron and timestep, with ``r = 1 if mem > threshold else 0`` (the previous step's reset)::

    candidate = beta*mem + current - r*threshold      reset "subtract"
              = beta*(1 - r)*mem + current            reset "zero"
              = beta*mem + current                    reset "none"
    candidate = Q(candidate)                          if state_quant is set
    fired     = 1 if candidate > threshold else 0

    reset_delay=True or reset "none":  new_mem = candidate
    reset_delay=False:                 d = fired - r
                                       new_mem = candidate - d*threshold   ("subtract")
                                       new_mem = candidate - d*candidate   ("zero")

    spike = fired * graded_spikes_factor

``graded_spikes_factor`` only scales the emitted spike; the reset arithmetic uses the binary
decision. With ``inhibition=True`` the layer is updated as a whole: only the neuron with the
largest ``candidate - threshold`` margin (lowest index on a tie) may fire, the emitted spike is
not scaled by ``graded_spikes_factor``, and the immediate correction uses
``d = spike / graded_spikes_factor - r``. Both quirks reproduce snnTorch 1.0.0 exactly.
"""

from typing import List, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F

from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel

State = List[torch.Tensor]  # one membrane tensor per LIF layer, in IR order


class StepTrace(NamedTuple):
    """Per-layer values of one timestep: affine outputs, emitted spikes, updated membranes."""

    currents: List[torch.Tensor]
    spikes: List[torch.Tensor]
    mem: List[torch.Tensor]


def initialize_state(ir: SNNModel, batch_size: int) -> State:
    """One zero membrane of shape ``(batch_size, N)`` per LIF layer."""
    state: State = []
    out_features = None
    for layer in ir.layers:
        if isinstance(layer, LinearLayer):
            out_features = layer.out_features
        elif isinstance(layer, LIFLayer):
            state.append(torch.zeros(batch_size, out_features))
    return state


def _quantize_state(candidate: torch.Tensor, levels: Tuple[float, ...]) -> torch.Tensor:
    """Nearest level, lowest index on a tie (``StateQuant.forward``)."""
    levels_t = torch.tensor(levels, dtype=candidate.dtype, device=candidate.device)
    idx = torch.min(torch.abs(levels_t - candidate.unsqueeze(-1)), dim=-1)[1]
    return levels_t[idx]


def _param(values, like: torch.Tensor) -> torch.Tensor:
    """A length-1 or length-N parameter list as a tensor that broadcasts against ``(batch, N)``."""
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def _candidate(layer: LIFLayer, current: torch.Tensor, mem: torch.Tensor):
    """Returns ``(candidate, reset_old, threshold)``, quantized if the layer quantizes state."""
    beta, threshold = _param(layer.beta, mem), _param(layer.threshold, mem)
    reset_old = (mem - threshold > 0).float()
    if layer.reset_mechanism == "subtract":
        candidate = beta * mem + current - reset_old * threshold
    elif layer.reset_mechanism == "zero":
        candidate = beta * ((1.0 - reset_old) * mem) + current
    elif layer.reset_mechanism == "none":
        candidate = beta * mem + current
    else:
        raise ValueError(f"unsupported reset_mechanism {layer.reset_mechanism!r} on layer {layer.name!r}")
    if layer.state_quant is not None:
        candidate = _quantize_state(candidate, layer.state_quant.levels)
    return candidate, reset_old, threshold


def _apply_reset(layer: LIFLayer, candidate, threshold, do_reset) -> torch.Tensor:
    if layer.reset_delay or layer.reset_mechanism == "none":
        return candidate
    if layer.reset_mechanism == "subtract":
        return candidate - do_reset * threshold
    return candidate - do_reset * candidate


def _lif_step(layer: LIFLayer, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """One LIF update; returns ``(emitted_spike, new_membrane)``."""
    candidate, reset_old, threshold = _candidate(layer, current, mem)
    graded = _param(layer.graded_spikes_factor, mem)
    fired = (candidate - threshold > 0).float()
    if layer.inhibition:
        margin = candidate - threshold
        winner = torch.zeros_like(fired)
        winner[torch.arange(mem.shape[0]), torch.argmax(margin, dim=1)] = 1.0
        spike = fired * winner
        return spike, _apply_reset(layer, candidate, threshold, spike / graded - reset_old)
    return fired * graded, _apply_reset(layer, candidate, threshold, fired - reset_old)


def step(ir: SNNModel, x: torch.Tensor, state: State) -> Tuple[torch.Tensor, State, StepTrace]:
    """Runs one timestep on ``x`` (``(batch, input_dim)``).

    Returns ``(output_spikes, new_state, trace)``.
    """
    currents, spikes, mems = [], [], []
    value = x
    for layer in ir.layers:
        if isinstance(layer, LinearLayer):
            value = F.linear(value, layer.weight, layer.bias)
            currents.append(value)
        else:
            value, mem = _lif_step(layer, value, state[len(mems)])
            spikes.append(value)
            mems.append(mem)
    return value, mems, StepTrace(currents, spikes, mems)


def run_sequence(
    ir: SNNModel, inputs: Sequence[torch.Tensor], state: State = None
) -> Tuple[List[torch.Tensor], State, List[StepTrace]]:
    """Runs ``step`` over one input per timestep, starting from zero state unless given.

    Returns ``(outputs, final_state, traces)``.
    """
    if state is None:
        state = initialize_state(ir, batch_size=inputs[0].shape[0])
    outputs, traces = [], []
    for x in inputs:
        output, state, trace = step(ir, x, state)
        outputs.append(output)
        traces.append(trace)
    return outputs, state, traces
