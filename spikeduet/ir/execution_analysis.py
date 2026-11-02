"""Read-only analyses over the Execution IR: def-use queries and temporal invariance."""


from dataclasses import dataclass


from typing import Dict, List, Optional, Set, Tuple


from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    DelayPushOp,
    ExecutionModule,
    LIFUpdateOp,
    TemporalMode,
    op_runtime_inputs,
)


def producers(module: ExecutionModule) -> Dict[str, str]:
    """Maps each buffer name to the id of the op that produces it within
    the module's loop body, for every buffer that has one.

    The model's input buffer and every stateful (membrane) buffer have no
    entry (they are not "produced" by any op -- see
    `compiler/ir/execution.py::Buffer.stateful` and
    `compiler/ir/execution_verifier.py`'s def-before-use rule). A
    `LIFUpdateOp` "produces" its `output_spike`, not its `membrane` (the
    membrane buffer is pre-existing state it updates in place, not a
    fresh value it defines -- see that op's docstring).
    """
    result: Dict[str, str] = {}
    for op in module.walk_ops():
        if isinstance(op, ConvertOp):
            result[op.output] = op.id
        elif isinstance(op, AffineOp):
            result[op.output] = op.id
        elif isinstance(op, DelayPushOp):
            result[op.history] = op.id
        elif isinstance(op, LIFUpdateOp):
            result[op.output_spike] = op.id
    return result


def consumers(module: ExecutionModule) -> Dict[str, List[str]]:
    """Maps each buffer name to the ids of every op that reads it (in
    program order), within the module's loop body.

    A `LIFUpdateOp` reads both `current` and `membrane` -- both appear as
    consumers of that op's id.
    """
    result: Dict[str, List[str]] = {}

    def _add(name: str, op_id: str) -> None:
        result.setdefault(name, []).append(op_id)

    for op in module.walk_ops():
        if isinstance(op, ConvertOp):
            _add(op.input, op.id)
        elif isinstance(op, AffineOp):
            _add(op.input, op.id)
        elif isinstance(op, DelayPushOp):
            _add(op.input, op.id)
        elif isinstance(op, LIFUpdateOp):
            for name in op_runtime_inputs(op):     # current, membrane (+ spike_state / adapt when present)
                _add(name, op.id)
    return result


def single_consumer(module: ExecutionModule, buffer_name: str) -> Optional[str]:
    """Id of the only op reading ``buffer_name``, or ``None`` if there are zero or several."""
    cons = consumers(module).get(buffer_name, [])
    return cons[0] if len(cons) == 1 else None


def producer(module: ExecutionModule, buffer_name: str) -> Optional[str]:
    """Id of the op defining ``buffer_name``, or ``None``."""
    return producers(module).get(buffer_name)


def is_dead_after_removing_ops(
    module: ExecutionModule, buffer_name: str, removed_op_ids: Set[str]
) -> bool:
    """Whether `buffer_name` would have ZERO remaining consumers once
    every op id in `removed_op_ids` is deleted from `module` -- used to
    decide if a buffer itself can be safely dropped after a rewrite (B4
    Phase B2: `SpikeStagingFusionPass` deleting a `ConvertOp`). A narrow,
    single-rewrite dead-buffer check, not a general liveness/SSA
    framework -- see that module's docstring for why only buffers made
    dead DIRECTLY by the fusion rewrite are ever removed."""
    cons = consumers(module).get(buffer_name, [])
    return all(c in removed_op_ids for c in cons)


@dataclass(frozen=True)
class TemporalInvarianceAnalysis:
    """Read-only propagation of `TemporalMode.INVARIANT` through the
    module's dataflow, starting from `module.input_buffer`'s own recorded
    mode.

    Propagation rule (structural, not a guess -- re-derives exactly the
    legality condition `firmware/common/snn.h::iscas_layer_input_is_invariant`
    documents in prose): a `ConvertOp`'s or `AffineOp`'s output is
    invariant if and only if its input buffer is invariant (weights/bias
    are always compile-time constants, so an invariant input always
    produces an invariant affine drive). A `LIFUpdateOp`'s `output_spike`
    (and its `membrane`) is NEVER invariant, regardless of `current`'s
    invariance -- membrane state is recurrent (depends on the PREVIOUS
    timestep's own membrane value), so it varies even under a perfectly
    static input. This is why, under the currently supported single-chain
    `[Linear -> Leaky]^N` topology, AT MOST the first layer's `ConvertOp`+
    `AffineOp` pair can ever be invariant: every later layer's input is a
    spike buffer, which this analysis (correctly) never marks invariant.

    B4 Phase A note on `Buffer.temporal`:
        `Buffer.temporal` is seed/source temporal metadata set at lowering
        time (e.g. `TemporalMode.INVARIANT` for a static MNIST input).
        Derived invariance for ALL other buffers comes exclusively from
        this analysis, NOT from `Buffer.temporal` -- lowering sets every
        non-input buffer's `.temporal` to `VARYING` regardless of input
        mode. This analysis is the single source of truth for asking
        "is buffer X provably timestep-invariant?"

    Construct via `TemporalInvarianceAnalysis.compute(module)`. Does not
    mutate `module` or its buffers' own `.temporal` fields (those keep
    whatever `lower_source_ir` set, unchanged) -- this is a separately
    computed, queryable fact, not a rewrite.
    """

    invariant_buffers: Tuple[str, ...]
    varying_buffers: Tuple[str, ...]

    def is_invariant(self, buffer_name: str) -> bool:
        return buffer_name in self.invariant_buffers

    @staticmethod
    def compute(module: ExecutionModule) -> "TemporalInvarianceAnalysis":
        invariant = set()
        varying = set()

        # Seed from the external input's declared temporal mode.
        input_buf = module.get_buffer(module.input_buffer)
        if input_buf.temporal is TemporalMode.INVARIANT:
            invariant.add(module.input_buffer)
        else:
            varying.add(module.input_buffer)

        # Membrane (stateful) buffers are always varying -- recurrent
        # state depends on the previous timestep's own value.
        for b in module.buffers:
            if b.stateful:
                varying.add(b.name)

        # Propagate through all ops in program order (preheader + loop).
        for op in module.walk_ops():
            if isinstance(op, ConvertOp):
                if op.input in invariant:
                    invariant.add(op.output)
                else:
                    varying.add(op.output)
            elif isinstance(op, AffineOp):
                if op.input in invariant:
                    invariant.add(op.output)
                else:
                    varying.add(op.output)
            elif isinstance(op, DelayPushOp):
                varying.add(op.history)    # stateful ring: never invariant
            elif isinstance(op, LIFUpdateOp):
                # membrane already in `varying` (stateful, seeded above)
                varying.add(op.output_spike)

        return TemporalInvarianceAnalysis(
            invariant_buffers=tuple(sorted(invariant)),
            varying_buffers=tuple(sorted(varying)),
        )
