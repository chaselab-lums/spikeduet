"""Execution IR: how a verified source model runs.

The source IR (``spikeduet.ir.model``) says what a model computes. The Execution IR says how: an
explicit timestep loop, named logical buffers (input, staging, current, membrane, spike) and the
operations that connect them in program order. Passes rewrite it and backends emit code from it.

It deliberately has no physical addresses, no CPU/accelerator placement and no readout operation:
those are backend decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

from spikeduet.ir.model import LIFLayer, LinearLayer, StateQuantSpec


class TemporalMode(Enum):
    """Whether a buffer holds the same value on every timestep.

    The caller declares it for the model input (a static image held for all steps is
    ``INVARIANT``, an event stream is ``VARYING``); it is derived for every other buffer.
    Membrane and spike buffers are always ``VARYING`` because LIF state is recurrent.
    """

    INVARIANT = "invariant"
    VARYING = "varying"


class StorageKind(Enum):
    """Runtime storage type of a buffer. Lowering only produces ``F32``; an FP16 cast needed by
    an accelerator is part of that backend's realization of the affine op."""

    F32 = "f32"
    F16 = "f16"


class BufferKind(Enum):
    """Role of a buffer in the schedule."""

    INPUT = "input"
    STAGING = "staging"
    CURRENT = "current"
    MEMBRANE = "membrane"
    SPIKE = "spike"


@dataclass(frozen=True)
class Buffer:
    """A logical value, not a memory allocation.

    Attributes:
        name: unique identifier, e.g. ``"mem0"``.
        shape: ``(num_elements,)``; the runtime processes one sample at a time.
        dtype: storage type.
        kind: role in the schedule.
        stateful: ``True`` for membranes, which persist across iterations and are zeroed once
            before the loop; every other buffer is recomputed each iteration.
        temporal: timestep invariance (see ``TemporalMode``).

    Producers and consumers are not stored here; ``spikeduet.ir.execution_analysis`` computes
    them from the op sequence.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: StorageKind
    kind: BufferKind
    stateful: bool
    temporal: TemporalMode


@dataclass(frozen=True)
class LIFExecutionSpec:
    """Immutable copy of an ``LIFLayer``'s semantics plus its neuron count."""

    num_neurons: int
    beta: Tuple[float, ...]
    threshold: Tuple[float, ...]
    reset_mechanism: str
    reset_delay: bool
    graded_spikes_factor: Tuple[float, ...]
    inhibition: bool
    state_quant: Optional[StateQuantSpec]

    @staticmethod
    def from_source(lif: LIFLayer, num_neurons: int) -> "LIFExecutionSpec":
        return LIFExecutionSpec(
            num_neurons=num_neurons,
            beta=tuple(lif.beta),
            threshold=tuple(lif.threshold),
            reset_mechanism=lif.reset_mechanism,
            reset_delay=lif.reset_delay,
            graded_spikes_factor=tuple(lif.graded_spikes_factor),
            inhibition=lif.inhibition,
            state_quant=lif.state_quant,
        )


@dataclass(frozen=True)
class AffineOp:
    """``output = weight @ input + bias``. ``weight_layer`` references the source
    ``LinearLayer``; weights are never copied into the Execution IR."""

    id: str
    input: str
    weight_layer: LinearLayer
    output: str


@dataclass(frozen=True)
class LIFUpdateOp:
    """One timestep of an LIF layer: reads ``current`` and ``membrane``, writes the updated
    ``membrane`` and ``output_spike``."""

    id: str
    current: str
    membrane: str
    output_spike: str
    semantics: LIFExecutionSpec


@dataclass(frozen=True)
class ConvertOp:
    """Staging copy ``output = cast(input)`` in front of an affine.

    Lowering emits one before every ``AffineOp``. It is an identity copy today, and it is what
    ``SpikeStagingFusionPass`` removes when the upstream LIF can write the staging buffer
    directly.
    """

    id: str
    input: str
    output: str


ExecutionOp = Union[AffineOp, LIFUpdateOp, ConvertOp]


@dataclass(frozen=True)
class TemporalLoop:
    """A region executed ``trip_count`` times; ``body`` is one iteration in program order."""

    id: str
    trip_count: int
    body: Tuple[ExecutionOp, ...]


@dataclass(frozen=True)
class ExecutionModule:
    """A lowered model.

    Attributes:
        buffers: every logical buffer, in creation order.
        loop: the top-level temporal loop (a one-element tuple).
        num_timesteps: equal to ``loop[0].trip_count``.
        input_buffer: name of the external input buffer.
        output_buffer: name of the final layer's spike buffer.
        preheader_ops: ops that run once before the loop; empty after lowering and filled by
            passes such as ``InvariantAffineHoistingPass``.
    """

    buffers: Tuple[Buffer, ...]
    loop: Tuple[TemporalLoop, ...]
    num_timesteps: int
    input_buffer: str
    output_buffer: str
    preheader_ops: Tuple[ExecutionOp, ...] = ()

    def get_buffer(self, name: str) -> Buffer:
        for b in self.buffers:
            if b.name == name:
                return b
        raise KeyError(f"no buffer named {name!r} in this ExecutionModule")

    def walk_ops(self) -> Tuple[ExecutionOp, ...]:
        """Every op in program order: preheader first, then each loop body."""
        ops = list(self.preheader_ops)
        for region in self.loop:
            ops.extend(region.body)
        return tuple(ops)


def op_is_pure(op: ExecutionOp) -> bool:
    """Affine and convert ops depend only on their inputs; LIF updates carry state."""
    return isinstance(op, (AffineOp, ConvertOp))


def op_reads_state(op: ExecutionOp) -> bool:
    return isinstance(op, LIFUpdateOp)


def op_writes_state(op: ExecutionOp) -> bool:
    return isinstance(op, LIFUpdateOp)


def op_runtime_inputs(op: ExecutionOp) -> Tuple[str, ...]:
    """Buffers ``op`` reads at runtime (weights are compile-time constants)."""
    if isinstance(op, (ConvertOp, AffineOp)):
        return (op.input,)
    if isinstance(op, LIFUpdateOp):
        return (op.current, op.membrane)
    return ()


def op_outputs(op: ExecutionOp) -> Tuple[str, ...]:
    """Buffers ``op`` defines."""
    if isinstance(op, (ConvertOp, AffineOp)):
        return (op.output,)
    if isinstance(op, LIFUpdateOp):
        return (op.output_spike,)
    return ()
