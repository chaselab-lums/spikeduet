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
    """Returns one zero membrane tensor per LIF layer, shape (batch_size, N).

    Mirrors `snntorch.Leaky`'s own auto-initialization (an unset `mem`
    starts at shape-mismatched `zeros(0)` and is reinitialized to
    `zeros_like(input_)` on first use) -- membrane state always starts at
    zero for every neuron.
    """
    state: State = []
    out_features = None
    for layer in ir.layers:
        if isinstance(layer, LinearLayer):
            out_features = layer.out_features
        elif isinstance(layer, LIFLayer):
            state.append(torch.zeros(batch_size, out_features))
    # FilterBank extension state (empty for every ordinary model, so the
    # state list is then exactly the pre-extension per-LIF membrane list).
    # Appended in this order, each in IR order: a previous-spike tensor per
    # `spike_state` LIF layer, an adaptation tensor per ALIF layer, an input
    # history tensor per delayed Linear layer. `_extension_index` maps layers
    # to their slots.
    out_features = None
    for layer in ir.layers:
        if isinstance(layer, LinearLayer):
            out_features = layer.out_features
        elif isinstance(layer, LIFLayer) and layer.spike_state:
            state.append(torch.zeros(batch_size, out_features))
    for layer in ir.layers:
        if isinstance(layer, LinearLayer):
            out_features = layer.out_features
        elif isinstance(layer, LIFLayer) and layer.adaptation is not None:
            state.append(torch.zeros(batch_size, out_features))
    for layer in ir.layers:
        if isinstance(layer, LinearLayer) and layer.synapse_delays is not None:
            depth = int(layer.synapse_delays.max()) + 1
            state.append(torch.zeros(batch_size, depth, layer.in_features))
    return state


def _extension_index(ir: SNNModel):
    """Slot of each layer's extension state inside the state list:
    `(sprev, adapt, hist)` dicts keyed by position in `ir.layers`."""
    n_lif = sum(isinstance(l, LIFLayer) for l in ir.layers)
    sprev, adapt, hist = {}, {}, {}
    nxt = n_lif
    for pos, l in enumerate(ir.layers):
        if isinstance(l, LIFLayer) and l.spike_state:
            sprev[pos] = nxt; nxt += 1
    for pos, l in enumerate(ir.layers):
        if isinstance(l, LIFLayer) and l.adaptation is not None:
            adapt[pos] = nxt; nxt += 1
    for pos, l in enumerate(ir.layers):
        if isinstance(l, LinearLayer) and l.synapse_delays is not None:
            hist[pos] = nxt; nxt += 1
    return sprev, adapt, hist


def _lif_step_spike_state(layer: LIFLayer, current: torch.Tensor, mem: torch.Tensor, sprev: torch.Tensor,
                          adapt: torch.Tensor = None):
    """Delayed-subtract LIF keyed on the STORED previous spike (see
    `LIFLayer.spike_state`), optionally with an adaptive threshold (see
    `AdaptiveThresholdSpec`). Returns `(spike, new_mem, new_adapt)`
    (`new_adapt` is None without adaptation); the caller stores `spike` as
    the next `sprev`."""
    dev, dt = mem.device, mem.dtype
    beta = torch.tensor(layer.beta, dtype=dt, device=dev)
    threshold = torch.tensor(layer.threshold, dtype=dt, device=dev)
    if layer.adaptation is not None:
        rho = torch.tensor(layer.adaptation.rho, dtype=dt, device=dev)
        a = torch.tensor(layer.adaptation.a, dtype=dt, device=dev)
        new_adapt = rho * adapt + sprev
        th_eff = threshold + a * new_adapt
    else:
        new_adapt, th_eff = None, threshold
    new_mem = beta * mem + current - sprev * th_eff
    spike = (new_mem - th_eff > 0).float()
    return spike, new_mem, new_adapt


def _quantize_state(candidate: torch.Tensor, levels: Tuple[float, ...]) -> torch.Tensor:
    """L5: nearest-configured-level quantization, mirroring
    `snntorch.functional.quant.StateQuant.forward` exactly:
    `levels[argmin_i |levels[i] - x|]`, first index wins ties. `levels` is
    expected in the exact (ascending, verified) order the frontend
    extracted it in -- see `StateQuantSpec` and this module's docstring.
    """
    levels_t = torch.tensor(levels, dtype=candidate.dtype, device=candidate.device)
    diffs = torch.abs(levels_t - candidate.unsqueeze(-1))
    idx = torch.min(diffs, dim=-1)[1]
    return levels_t[idx]


def _lif_step_inhibition(layer: LIFLayer, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """L4: layer-wide winner-take-all LIF update -- see this module's
    docstring for the full source-re-audited derivation. `mem`/`current`
    are shape (batch, N); winner selection is independent per batch row.
    """
    beta = torch.tensor(layer.beta, dtype=mem.dtype, device=mem.device)
    threshold = torch.tensor(layer.threshold, dtype=mem.dtype, device=mem.device)
    graded_factor = torch.tensor(layer.graded_spikes_factor, dtype=mem.dtype, device=mem.device)

    reset_old = (mem - threshold > 0).float()

    if layer.reset_mechanism == "subtract":
        mem_state = beta * mem + current - reset_old * threshold
    elif layer.reset_mechanism == "zero":
        zeroed = (1.0 - reset_old) * mem
        mem_state = beta * zeroed + current
    elif layer.reset_mechanism == "none":
        mem_state = beta * mem + current
    else:
        raise ValueError(
            f"unsupported reset_mechanism {layer.reset_mechanism!r} on layer "
            f"{layer.name!r} -- compiler/frontend/snntorch_parser.py should "
            "have rejected this before it ever reached the IR"
        )

    # L5: quantize the candidate BEFORE margin/winner selection -- see
    # this module's docstring. Winner selection genuinely sees the
    # quantized membrane, not the raw one, structurally (matching
    # snnTorch's single `self.mem = self.state_quant(self.mem)`
    # reassignment before either fire()/fire_inhibition() runs).
    if layer.state_quant is not None:
        mem_state = _quantize_state(mem_state, layer.state_quant.levels)

    margin = mem_state - threshold
    fired = (margin > 0).float()

    # Winner = highest margin per batch row (torch.argmax picks the first
    # index among exact ties, in the tested torch version -- see docstring).
    winner_idx = torch.argmax(margin, dim=1)
    winner_mask = torch.zeros_like(fired)
    winner_mask[torch.arange(mem.shape[0]), winner_idx] = 1.0

    # The winner-masked, UNSCALED binary decision -- this is exactly
    # snnTorch's `spk` variable inside forward() when inhibition=True:
    # never multiplied by graded_spikes_factor (see docstring).
    spk = fired * winner_mask

    if not layer.reset_delay and layer.reset_mechanism in ("subtract", "zero"):
        do_reset = spk / graded_factor - reset_old
        if layer.reset_mechanism == "subtract":
            new_mem = mem_state - do_reset * threshold
        else:  # zero
            new_mem = mem_state - do_reset * mem_state
    else:
        new_mem = mem_state

    # Emitted value == spk itself, unscaled -- inhibition never applies
    # graded_spikes_factor to the returned/next-layer-input value.
    return spk, new_mem


def _lif_step(layer: LIFLayer, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer.inhibition:
        return _lif_step_inhibition(layer, current, mem)

    # L2: layer.beta/layer.threshold are always lists (length 1 = shared
    # scalar, length N = per-neuron -- see the L2 report and
    # compiler/ir/model.py::LIFLayer). Converting to a tensor of shape
    # (1,) or (N,) broadcasts correctly against mem's (batch, N) shape
    # either way via ordinary PyTorch broadcasting -- no separate
    # scalar/vector code path needed here.
    beta = torch.tensor(layer.beta, dtype=mem.dtype, device=mem.device)
    threshold = torch.tensor(layer.threshold, dtype=mem.dtype, device=mem.device)
    # L3: graded_spikes_factor scales only the value returned/traced as
    # "spikes" -- it never appears in any reset_old/fired/reset math below
    # (see this module's docstring).
    graded_factor = torch.tensor(layer.graded_spikes_factor, dtype=mem.dtype, device=mem.device)

    reset_old = (mem - threshold > 0).float()

    # L5: quantize whichever "candidate" value each branch below computes
    # BEFORE that branch's own firing decision -- see this module's
    # docstring. When `state_quant` is disabled this is a no-op function
    # call site that is simply never reached (`is not None` guards below),
    # so L1-L4 behavior is exactly unchanged.
    quantize = layer.state_quant is not None

    if layer.reset_mechanism == "subtract":
        if not layer.reset_delay:
            updated = beta * mem + current - reset_old * threshold
            if quantize:
                updated = _quantize_state(updated, layer.state_quant.levels)
            fired = (updated - threshold > 0).float()
            # L5: uses `updated` (the actual candidate, quantized or not)
            # directly rather than recomputing beta*mem+current from
            # scratch -- algebraically identical to the pre-L5 shortcut
            # when state_quant is disabled (do_reset=fired-reset_old
            # against `updated`, which already has reset_old*threshold
            # baked in -- the reset_old terms cancel exactly as in L1's
            # own proof), and the only well-defined form once `updated`
            # may have been quantized.
            do_reset = fired - reset_old
            new_mem = updated - do_reset * threshold
        else:
            new_mem = beta * mem + current - reset_old * threshold
            if quantize:
                new_mem = _quantize_state(new_mem, layer.state_quant.levels)
            fired = (new_mem - threshold > 0).float()

    elif layer.reset_mechanism == "zero":
        zeroed = (1.0 - reset_old) * mem
        if not layer.reset_delay:
            updated = beta * zeroed + current
            if quantize:
                updated = _quantize_state(updated, layer.state_quant.levels)
            fired = (updated - threshold > 0).float()
            do_reset = fired - reset_old
            new_mem = (1.0 - do_reset) * updated
        else:
            new_mem = beta * zeroed + current
            if quantize:
                new_mem = _quantize_state(new_mem, layer.state_quant.levels)
            fired = (new_mem - threshold > 0).float()

    elif layer.reset_mechanism == "none":
        # reset_delay is numerically irrelevant here -- no reset is ever
        # applied either way (matches snnTorch: forward()'s reset_delay
        # branch has no case for reset_mechanism_val==2).
        new_mem = beta * mem + current
        if quantize:
            new_mem = _quantize_state(new_mem, layer.state_quant.levels)
        fired = (new_mem - threshold > 0).float()

    else:
        raise ValueError(
            f"unsupported reset_mechanism {layer.reset_mechanism!r} on layer "
            f"{layer.name!r} -- compiler/frontend/snntorch_parser.py should "
            "have rejected this before it ever reached the IR"
        )

    # L3: the emitted spike (what this function returns, and what feeds the
    # next layer) is the binary firing decision scaled by graded_spikes_factor
    # -- `fired` itself (unscaled) has already been fully consumed above by
    # every reset computation that needed it.
    emitted = fired * graded_factor
    return emitted, new_mem


def step(ir: SNNModel, x: torch.Tensor, state: State) -> Tuple[torch.Tensor, State, StepTrace]:
    """Executes exactly one SNN timestep of `ir` on input `x`.

    Args:
        ir: The parsed `[Linear -> Leaky]^N` model.
        x: Input for this timestep, shape (batch_size, ir input dim).
        state: Membrane state from `initialize_state` or a prior `step`
            call, one tensor per LIF layer in IR order.

    Returns:
        A tuple `(output, new_state, trace)`: `output` is the spike output
        of the final LIF layer; `new_state` is the updated per-layer
        membrane state; `trace` exposes every layer's intermediate values.
    """
    trace_currents: List[torch.Tensor] = []
    trace_spikes: List[torch.Tensor] = []
    trace_mem: List[torch.Tensor] = []
    sprev_ix, adapt_ix, hist_ix = _extension_index(ir)
    new_state: State = list(state)            # extension state is updated in place below

    value = x
    lif_index = 0
    for pos, layer in enumerate(ir.layers):
        if isinstance(layer, LinearLayer):
            if layer.synapse_delays is not None:
                # push this step's input into the history ring (slot d = d steps ago), then
                # gather each synapse's delayed input: out[b, j, i] = hist[b, delay[j, i], i]
                hist = torch.cat([value.unsqueeze(1), state[hist_ix[pos]][:, :-1]], dim=1)
                new_state[hist_ix[pos]] = hist
                col = torch.arange(layer.in_features)
                gathered = hist[:, layer.synapse_delays, col]
                value = (gathered * layer.weight).sum(-1)
                if layer.bias is not None:
                    value = value + layer.bias
            else:
                value = F.linear(value, layer.weight, layer.bias)
            trace_currents.append(value)
        else:  # LIFLayer
            if layer.spike_state:
                adapt = state[adapt_ix[pos]] if pos in adapt_ix else None
                spk, mem, new_adapt = _lif_step_spike_state(layer, value, state[lif_index], state[sprev_ix[pos]], adapt)
                new_state[sprev_ix[pos]] = spk
                if pos in adapt_ix:
                    new_state[adapt_ix[pos]] = new_adapt
            else:
                spk, mem = _lif_step(layer, value, state[lif_index])
            new_state[lif_index] = mem
            trace_spikes.append(spk)
            trace_mem.append(mem)
            value = spk
            lif_index += 1

    return value, new_state, StepTrace(trace_currents, trace_spikes, trace_mem)


def run_sequence(
    ir: SNNModel, inputs: Sequence[torch.Tensor], state: State = None
) -> Tuple[List[torch.Tensor], State, List[StepTrace]]:
    """Runs `step` over a sequence of per-timestep inputs.

    Args:
        ir: The parsed `[Linear -> Leaky]^N` model.
        inputs: One input tensor per timestep, each shape (batch_size, ir
            input dim).
        state: Initial membrane state; if omitted, a fresh zero state is
            created via `initialize_state` using the batch size of the
            first input.

    Returns:
        A tuple `(outputs, final_state, traces)`: `outputs[t]` and
        `traces[t]` are this module's `step` return values for timestep
        `t`; `final_state` is the membrane state after the last timestep.
    """
    if state is None:
        state = initialize_state(ir, batch_size=inputs[0].shape[0])

    outputs: List[torch.Tensor] = []
    traces: List[StepTrace] = []
    for x in inputs:
        output, state, trace = step(ir, x, state)
        outputs.append(output)
        traces.append(trace)

    return outputs, state, traces


def readout_scores(ir: SNNModel, traces: Sequence[StepTrace]) -> torch.Tensor:
    """The deployed readout, from a `run_sequence` trace list: the final LIF
    layer's membrane summed over all timesteps -- or, when
    `ir.readout_weights` is set, weighted per timestep
    (`sum_t w[t] * mem_out[t]`). Shape (batch, out_features); the predicted
    class is its argmax. Mirrors what every backend's generated readout
    computes."""
    mems = torch.stack([t.mem[-1] for t in traces], dim=0)          # [T, B, N]
    if ir.readout_weights is None:
        return mems.sum(0)
    w = torch.tensor(ir.readout_weights, dtype=mems.dtype)
    if w.shape[0] != mems.shape[0]:
        raise ValueError(
            f"readout_weights has {w.shape[0]} entries but the sequence has {mems.shape[0]} timesteps"
        )
    return (w[:, None, None] * mems).sum(0)


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
