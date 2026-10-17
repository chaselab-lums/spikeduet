"""Read-only analyses over the Execution IR: def-use queries and temporal invariance."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from spikeduet.ir.execution import (
    ExecutionModule,
    LIFUpdateOp,
    TemporalMode,
    op_outputs,
    op_runtime_inputs,
)


def producers(module: ExecutionModule) -> Dict[str, str]:
    """Buffer name -> id of the op that defines it.

    The model input and the membranes have no producer: a ``LIFUpdateOp`` defines its spike
    output and updates its membrane in place.
    """
    return {out: op.id for op in module.walk_ops() for out in op_outputs(op)}


def consumers(module: ExecutionModule) -> Dict[str, List[str]]:
    """Buffer name -> ids of the ops that read it, in program order."""
    result: Dict[str, List[str]] = {}
    for op in module.walk_ops():
        for name in op_runtime_inputs(op):
            result.setdefault(name, []).append(op.id)
    return result


def single_consumer(module: ExecutionModule, buffer_name: str) -> Optional[str]:
    """Id of the only op reading ``buffer_name``, or ``None`` if there are zero or several."""
    cons = consumers(module).get(buffer_name, [])
    return cons[0] if len(cons) == 1 else None


def producer(module: ExecutionModule, buffer_name: str) -> Optional[str]:
    """Id of the op defining ``buffer_name``, or ``None``."""
    return producers(module).get(buffer_name)


def is_dead_after_removing_ops(module: ExecutionModule, buffer_name: str, removed_op_ids: Set[str]) -> bool:
    """Whether every reader of ``buffer_name`` is in ``removed_op_ids``."""
    return all(c in removed_op_ids for c in consumers(module).get(buffer_name, []))


@dataclass(frozen=True)
class TemporalInvarianceAnalysis:
    """Which buffers hold the same value on every timestep.

    Starts from the input buffer's declared mode. A convert or affine output is invariant iff
    its input is (weights are constants). LIF outputs and membranes are never invariant, because
    membrane state is recurrent. In a single chain this means at most the first layer's convert
    and affine can be invariant. ``Buffer.temporal`` is only the seed; query this analysis for
    every derived buffer.
    """

    invariant_buffers: Tuple[str, ...]
    varying_buffers: Tuple[str, ...]

    def is_invariant(self, buffer_name: str) -> bool:
        return buffer_name in self.invariant_buffers

    @staticmethod
    def compute(module: ExecutionModule) -> "TemporalInvarianceAnalysis":
        invariant, varying = set(), set()
        if module.get_buffer(module.input_buffer).temporal is TemporalMode.INVARIANT:
            invariant.add(module.input_buffer)
        else:
            varying.add(module.input_buffer)
        varying.update(b.name for b in module.buffers if b.stateful)

        for op in module.walk_ops():
            for out in op_outputs(op):
                if not isinstance(op, LIFUpdateOp) and op.input in invariant:
                    invariant.add(out)
                else:
                    varying.add(out)

        return TemporalInvarianceAnalysis(
            invariant_buffers=tuple(sorted(invariant)),
            varying_buffers=tuple(sorted(varying)),
        )
