"""Source IR: an ordered ``[Linear -> LIF]^N`` chain with fully resolved parameters."""


from dataclasses import dataclass, field


from typing import List, Optional, Tuple


import torch


@dataclass(frozen=True)
class StateQuantSpec:
    """Membrane quantization to a fixed, ascending set of levels.

    ``Q(x) = levels[argmin_i |levels[i] - x|]``; on an exact tie the lower level wins, matching
    ``snntorch.functional.quant.state_quant``.
    """

    levels: Tuple[float, ...]


@dataclass(frozen=True)
class AdaptiveThresholdSpec:
    """Adaptive-threshold (ALIF) extension of a `LIFLayer` -- added for the
    spike-sorting FilterBank workload (see `compiler/frontend/filterbank_parser.py`).

    Each neuron carries a second state ``A`` (adaptation), updated every
    timestep from the PREVIOUS timestep's spike and used to raise the
    firing threshold (``s_prev`` is the layer's stored previous spike, see
    ``LIFLayer.spike_state``, which adaptation requires):

        A         = rho*A + s_prev                         (+1.0 per spike, then leak)
        th_eff    = threshold + a*A
        mem       = beta*mem + current - s_prev*th_eff     (subtract, delayed one step)
        spike     = 1 if mem > th_eff else 0

    ``rho``/``a`` follow the beta/threshold convention: length 1 (shared)
    or one value per neuron; ``rho`` in [0, 1], ``a`` >= 0 (the source
    model's ``|a|`` is resolved at export time). Only legal on a layer with
    ``reset_mechanism="subtract"``, ``reset_delay=True``, no inhibition, no
    state quantization and unit graded_spikes_factor -- enforced by
    `compiler/ir/verifier.py` (anything else would need a separately
    derived recurrence, not a silent approximation).
    """

    rho: Tuple[float, ...]
    a: Tuple[float, ...]


@dataclass
class LinearLayer:
    """A fully-connected layer.

    Attributes:
        name: Module name/path as discovered in the source model.
        in_features: Input dimension.
        out_features: Output dimension.
        weight: Weight tensor of shape (out_features, in_features), CPU, detached.
        bias: Optional bias tensor of shape (out_features,), CPU, detached.
        synapse_delays: Optional integer tensor of shape (out_features,
            in_features), CPU, values in [0, D]. When set, synapse (j, i)
            reads input element i as it was ``synapse_delays[j, i]``
            timesteps ago (input before t=0 is 0):

                current[j] = bias[j] + sum_i weight[j, i] * x_i[t - synapse_delays[j, i]]

            ``None`` (the default, every pre-FilterBank model) is the plain
            undelayed affine. Fixed per-synapse delays are a structural
            property of the layer, not trainable state.
    """

    name: str
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    synapse_delays: Optional[torch.Tensor] = None


@dataclass
class LIFLayer:
    """A leaky integrate-and-fire neuron layer.

    Attributes:
        name: Module name/path as discovered in the source model.
        beta: Membrane decay rate(s), ALWAYS a list (L2 -- see the L2
            report): length 1 means one shared scalar for every neuron in
            the layer, length `out_features` means one independent value
            per neuron. This is the single representation for both cases
            (no separate scalar/vector type) so every consumer -- IR,
            reference simulator, codegen, runtime -- has exactly one shape
            to handle, distinguished only by `len(beta)`. Values are
            already clamped to `[0, 1]` here (matching snnTorch's own
            `beta.clamp(0, 1)` inside `Leaky._base_state_function`,
            applied on every forward call) -- the compiler clamps ONCE at
            export time rather than the runtime clamping every timestep,
            which is exactly equivalent because a frozen, never-retrained
            beta never changes after export and `clamp` is idempotent
            (`clamp(clamp(x)) == clamp(x)`); see
            `compiler/frontend/snntorch_parser.py` for the rationale.
        threshold: Firing threshold(s), same length convention as `beta`.
            snnTorch does not clamp threshold, so these are the raw
            resolved values.
        reset_mechanism: One of snnTorch's reset modes: "subtract", "zero", or "none".
        reset_delay: Whether the reset is deferred one timestep (snnTorch's
            own default, `True`) or applied immediately in the same
            timestep as the spike that triggered it (`False`, the legacy
            EFR32 firmware's behavior). Numerically irrelevant when
            `reset_mechanism == "none"` (no reset is ever applied either
            way), but the source model's actual value is preserved here
            rather than silently normalized/rewritten -- see the L1 Leaky
            semantics audit report.
        graded_spikes_factor: Output-spike scale factor(s), ALWAYS a list
            (L3 -- see the L3 report), same length convention as `beta`:
            length 1 = one shared scalar, length `out_features` =
            per-neuron. Defaults to `[1.0]` (snnTorch's own default,
            unscaled binary spikes). Scales only the EMITTED spike value
            (what flows to the next layer / off-chip); the internal binary
            firing decision that drives reset math is a separate concept
            and is never scaled -- see `compiler/reference/simulator.py`'s
            module docstring and the L3 report for the snnTorch-source
            derivation of this distinction. NOTE (L4): this scaling is
            APPLIED ONLY WHEN `inhibition` is False -- see `inhibition`
            below and the L4 report's source re-audit:
            `SpikingNeuron.fire_inhibition()` never multiplies its output
            by `graded_spikes_factor` at all (unlike `fire()`), a real
            snnTorch-1.0.0-specific asymmetry this IR/runtime must
            reproduce exactly, not "fix".
        inhibition: Whether this layer uses snnTorch's layer-wide
            winner-take-all competition (L4 -- see the L4 report). `False`
            (the default, and the only value V1 supported through L3)
            means every neuron updates independently, exactly as in
            L1/L2/L3. `True` means only the single neuron with the
            highest `(membrane - threshold)` margin in the whole layer
            (per batch item) may emit a nonzero spike this timestep --
            every other neuron is forced to zero regardless of its own
            threshold crossing. This is a boolean-only feature: snnTorch
            1.0.0 exposes no further inhibition configuration (no groups,
            no partial competition), so none is invented here.
        state_quant: Resolved membrane-state quantization (L5 -- see the L5
            report), or `None` if disabled (the default, and the only
            value V1 supported through L4). When set, the state_function's
            candidate membrane is quantized to the nearest configured
            level BEFORE the firing decision (and, for `inhibition=True`,
            before winner selection) -- see `StateQuantSpec` and
            `compiler/reference/simulator.py`'s module docstring for the
            exact source-re-audited placement. The immediate-reset
            (`reset_delay=False`) correction is computed FROM the
            quantized candidate but its own output is never re-quantized
            -- confirmed empirically against live snnTorch, not assumed.
        adaptation: Optional adaptive-threshold (ALIF) state, see
            `AdaptiveThresholdSpec`; `None` (default) is an ordinary
            fixed-threshold neuron.
        spike_state: `False` (default, snnTorch's own semantics) derives the
            delayed-reset indicator from the membrane (`mem > threshold`,
            see `compiler/reference/simulator.py`). `True` stores the
            previous timestep's binary spike as explicit per-neuron state
            (zero at the start of an inference) and keys the delayed
            subtract reset on that. The two are identical from the second
            timestep on; they differ at t=0 whenever a threshold is
            NEGATIVE (a membrane at rest, 0, is then "above threshold", so
            the derived indicator spuriously resets while the stored one is
            correctly 0). Models trained with an explicit spike state
            (the FilterBank spike sorter has negative thresholds) need
            this to be reproduced exactly. Only legal with
            reset_mechanism='subtract', reset_delay=True, no inhibition, no
            state_quant, unit graded_spikes_factor (verifier-enforced).
    """

    name: str
    beta: List[float]
    threshold: List[float]
    reset_mechanism: str
    reset_delay: bool = False
    graded_spikes_factor: List[float] = field(default_factory=lambda: [1.0])
    inhibition: bool = False
    state_quant: Optional[StateQuantSpec] = None
    adaptation: Optional[AdaptiveThresholdSpec] = None
    spike_state: bool = False


@dataclass
class SNNModel:
    """An ordered sequential SNN model.

    Attributes:
        layers: Ordered list of layers, alternating Linear/LIF for the
            currently supported subset.
        num_steps: Optional number of simulation time steps, if known.
        readout: Optional name of a readout/decoding scheme (e.g. "rate").
        readout_weights: Optional per-timestep readout weights. ``None`` (the
            default, every pre-FilterBank model) means the standard readout:
            sum the final layer's membrane over all timesteps, argmax. When
            set, the readout is the WEIGHTED sum ``sum_t w[t] * mem_out[t]``
            (length must equal the deployed timestep count -- checked at
            lowering), still followed by argmax.
    """

    layers: list
    num_steps: Optional[int] = None
    readout: Optional[str] = None
    readout_weights: Optional[Tuple[float, ...]] = None
