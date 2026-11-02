"""Execution IR: how a verified source model runs.

The source IR (``spikeduet.ir.model``) says what a model computes. The Execution IR says how: an
explicit timestep loop, named logical buffers (input, staging, current, membrane, spike) and the
operations that connect them in program order. Passes rewrite it and backends emit code from it.

It deliberately has no physical addresses, no CPU/accelerator placement and no readout operation:
those are backend decisions.
"""


from dataclasses import dataclass, field


from enum import Enum


from typing import Optional, Tuple, Union


from spikeduet.ir.model import AdaptiveThresholdSpec, LIFLayer, LinearLayer, StateQuantSpec


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
    """What role a logical buffer plays in the execution schedule."""

    INPUT = "input"
    STAGING = "staging"
    CURRENT = "current"
    MEMBRANE = "membrane"
    SPIKE = "spike"
    # FilterBank extension buffers -- all stateful (zero-initialised once
    # per inference, persisting across the temporal loop):
    HISTORY = "history"          # input history ring of a delayed Linear layer, [(D+1)*in]
    ADAPTATION = "adaptation"    # ALIF adaptation state A of a LIF layer
    SPIKE_STATE = "spike_state"  # a LIF layer's stored previous binary spike


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
    """Immutable snapshot of a `LIFLayer`'s snnTorch-faithful semantics,
    for use inside `LIFUpdateOp`.

    Part 7 of the B2 spec: "Source IR says WHAT the neuron means, Execution
    IR says HOW that operation participates in the execution schedule."
    This type exists so `LIFUpdateOp` never re-derives or re-implements
    L1-L5 neuron semantics -- it is a plain, immutable copy of the exact
    fields `compiler/ir/model.py::LIFLayer` already carries (beta,
    threshold, reset_mechanism, reset_delay, graded_spikes_factor,
    inhibition, state_quant), PLUS `num_neurons` -- the one piece of
    information a `LIFLayer` does not itself store (source IR derives
    neuron count structurally, from the preceding `LinearLayer.out_features`;
    see `compiler/ir/verifier.py`), made explicit here so `LIFUpdateOp` and
    `verify_execution_ir` never need to cross-reference the source IR to
    know a buffer's expected size.

    Construct via `LIFExecutionSpec.from_source`, never by hand-copying
    fields -- keeps this the ONE place a `LIFLayer` is translated into its
    execution-time counterpart.
    """

    num_neurons: int
    beta: Tuple[float, ...]
    threshold: Tuple[float, ...]
    reset_mechanism: str
    reset_delay: bool
    graded_spikes_factor: Tuple[float, ...]
    inhibition: bool
    state_quant: Optional[StateQuantSpec]
    adaptation: Optional[AdaptiveThresholdSpec] = None
    spike_state: bool = False

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
            adaptation=lif.adaptation,
            spike_state=lif.spike_state,
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
    """One LIF layer's per-timestep recurrence.

    `membrane` is read (the incoming state) AND written (the updated
    state) by this op -- see `Buffer.stateful`. `current`/`output_spike`
    are plain per-iteration values. `semantics` is the immutable,
    source-derived `LIFExecutionSpec` this op must implement exactly (see
    that type's docstring) -- this op never stores its own copy of
    beta/threshold/etc. separately from `semantics`.
    """

    id: str
    current: str
    membrane: str
    output_spike: str
    semantics: LIFExecutionSpec
    # FilterBank extension state buffers (None for every ordinary layer):
    # `spike_state` holds the previous timestep's binary spike when
    # `semantics.spike_state`; `adapt` holds the ALIF adaptation state when
    # `semantics.adaptation` is set. Both are stateful buffers this op reads
    # AND updates in place, exactly like `membrane`.
    spike_state: Optional[str] = None
    adapt: Optional[str] = None


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


@dataclass(frozen=True)
class DelayPushOp:
    """Pushes this timestep's input vector into a delay-history ring
    (FilterBank extension, see `LinearLayer.synapse_delays`).

    `history` is a stateful buffer of `(depth + 1) * n` floats laid out
    `history[d * n + i]` = element `i` of the input `d` timesteps ago; this
    op shifts every slot one step older and writes `input` into slot 0. The
    delayed `AffineOp` reading `history` then gathers each synapse's own
    slot. `depth` is the maximum synapse delay `D`. Stateful and impure:
    never hoistable, never fused.
    """

    id: str
    input: str
    history: str
    depth: int


ExecutionOp = Union[AffineOp, LIFUpdateOp, ConvertOp, DelayPushOp]


@dataclass(frozen=True)
class TemporalLoop:
    """A region executed ``trip_count`` times; ``body`` is one iteration in program order."""

    id: str
    trip_count: int
    body: Tuple[ExecutionOp, ...]


@dataclass(frozen=True)
class ExecutionModule:
    """A fully lowered, executable `[Linear -> Leaky]^N` model.

    Attributes:
        buffers: Every logical buffer in the module, in a fixed
            (creation) order -- used as pretty-printing/iteration order,
            not a legality constraint. Look up by name via `get_buffer`.
        loop: The model's single top-level `TemporalLoop`. A tuple of
            length 1 in B2 (kept as a tuple, not a bare field, so a future
            phase that needs pre-loop or post-loop top-level ops -- e.g. a
            hoisted invariant affine, B4 -- has somewhere to put them
            without changing this field's type).
        num_timesteps: Same value as `loop.trip_count`, exposed at the
            module level for convenience (pretty-printing, tests).
        input_buffer: Name of the model's external input buffer.
        output_buffer: Name of the final layer's spike buffer -- the
            model's output. (Readout/argmax is a runtime policy, not
            represented here -- see this module's docstring.)
        preheader_ops: Operations that execute exactly once, BEFORE the
            temporal loop (B4 Phase A). Empty after lowering; populated
            only by explicit compiler passes such as
            `InvariantAffineHoistingPass`. Structural placement: an op
            in `preheader_ops` is definitionally outside the temporal
            loop -- no `hoisted=True` metadata flag or runtime predicate.
            Placed last (with a default of `()`) so all existing
            constructor calls remain compatible.
        readout_weights: Per-timestep weights of the readout (length
            `num_timesteps`), or `None` for the standard unweighted
            sum of the final membrane over time. Carried through from
            `SNNModel.readout_weights`; the readout itself remains a
            backend-generated policy (accumulate, then argmax).
    """

    buffers: Tuple[Buffer, ...]
    loop: Tuple[TemporalLoop, ...]
    num_timesteps: int
    input_buffer: str
    output_buffer: str
    preheader_ops: Tuple[ExecutionOp, ...] = ()
    readout_weights: Optional[Tuple[float, ...]] = None

    def get_buffer(self, name: str) -> Buffer:
        for b in self.buffers:
            if b.name == name:
                return b
        raise KeyError(f"no buffer named {name!r} in this ExecutionModule")

    def walk_ops(self) -> Tuple[ExecutionOp, ...]:
        """Every op in the module, in program order: preheader ops first
        (B4), then flattened across all temporal-loop bodies."""
        ops = list(self.preheader_ops)
        for region in self.loop:
            ops.extend(region.body)
        return tuple(ops)


def op_is_pure(op: ExecutionOp) -> bool:
    """Affine and convert ops depend only on their inputs; LIF updates carry state."""
    return isinstance(op, (AffineOp, ConvertOp))


def op_reads_state(op: ExecutionOp) -> bool:
    """Whether `op` reads recurrent/stateful buffer(s) (membrane)."""
    return isinstance(op, (LIFUpdateOp, DelayPushOp))


def op_writes_state(op: ExecutionOp) -> bool:
    """Whether `op` writes recurrent/stateful buffer(s) (membrane)."""
    return isinstance(op, (LIFUpdateOp, DelayPushOp))


def op_runtime_inputs(op: ExecutionOp) -> Tuple[str, ...]:
    """Buffer names `op` reads at runtime (excludes compile-time
    constants like `AffineOp.weight_layer`)."""
    if isinstance(op, ConvertOp):
        return (op.input,)
    elif isinstance(op, AffineOp):
        return (op.input,)
    elif isinstance(op, DelayPushOp):
        return (op.input, op.history)
    elif isinstance(op, LIFUpdateOp):
        return tuple(b for b in (op.current, op.membrane, op.spike_state, op.adapt) if b is not None)
    return ()


def op_outputs(op: ExecutionOp) -> Tuple[str, ...]:
    """Buffer names `op` produces/defines."""
    if isinstance(op, ConvertOp):
        return (op.output,)
    elif isinstance(op, AffineOp):
        return (op.output,)
    elif isinstance(op, DelayPushOp):
        return (op.history,)
    elif isinstance(op, LIFUpdateOp):
        return (op.output_spike,)
    return ()
