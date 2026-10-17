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


@dataclass
class LinearLayer:
    """A fully connected layer. ``weight`` is ``(out_features, in_features)``, ``bias`` is
    ``(out_features,)`` or ``None``; both are detached CPU tensors."""

    name: str
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None


@dataclass
class LIFLayer:
    """A leaky integrate-and-fire layer (the semantics of ``snntorch.Leaky``).

    ``beta``, ``threshold`` and ``graded_spikes_factor`` are always lists: length 1 is one value
    shared by every neuron, length ``out_features`` is one value per neuron. ``beta`` is already
    clamped to ``[0, 1]``.

    Attributes:
        reset_mechanism: ``"subtract"``, ``"zero"`` or ``"none"``.
        reset_delay: ``True`` applies the reset one step after the spike (snnTorch's default),
            ``False`` in the same step.
        graded_spikes_factor: scales the emitted spike only, never the reset arithmetic. It is
            ignored when ``inhibition`` is set, as in snnTorch.
        inhibition: layer-wide winner-take-all: only the neuron with the largest
            ``membrane - threshold`` margin may spike in a step.
        state_quant: optional membrane quantization, applied to the candidate membrane before
            the firing decision.
    """

    name: str
    beta: List[float]
    threshold: List[float]
    reset_mechanism: str
    reset_delay: bool = False
    graded_spikes_factor: List[float] = field(default_factory=lambda: [1.0])
    inhibition: bool = False
    state_quant: Optional[StateQuantSpec] = None


@dataclass
class SNNModel:
    """An ordered model: ``layers`` alternates ``LinearLayer`` and ``LIFLayer``.

    ``num_steps`` is the default number of timesteps used by lowering when the caller does not
    give one. ``readout`` is informational only.
    """

    layers: list
    num_steps: Optional[int] = None
    readout: Optional[str] = None
