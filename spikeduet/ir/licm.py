"""Temporal loop-invariant code motion.

``InvariantAffineHoistingPass`` moves every timestep-invariant ``AffineOp`` (and the convert
ops it depends on) from the temporal loop into the module preheader, so it runs once per
inference instead of once per step. With a static input this hoists the first layer's affine;
with a time-varying input nothing is hoistable and the module is returned unchanged.
"""


import dataclasses


from dataclasses import dataclass, field


from typing import Dict, List, Set, Tuple


from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    ExecutionModule,
    ExecutionOp,
    LIFUpdateOp,
    TemporalLoop,
    op_is_pure,
    op_outputs,
    op_runtime_inputs,
)


from spikeduet.ir.execution_analysis import TemporalInvarianceAnalysis


from spikeduet.ir.pass_manager import CompilerPass


def is_hoistable(op: ExecutionOp, analysis: TemporalInvarianceAnalysis) -> bool:
    """Whether `op` is LEGALLY hoistable out of a temporal loop.

    Legality requires all of:
      1. op is pure (no stateful side effects)
      2. all runtime inputs are temporally invariant
         (weights/biases are compile-time constants, not checked here)

    Does NOT consider profitability -- see `should_hoist`.
    """
    if not op_is_pure(op):
        return False

    for input_name in op_runtime_inputs(op):
        if not analysis.is_invariant(input_name):
            return False

    return True


def should_hoist(op: ExecutionOp, analysis: TemporalInvarianceAnalysis) -> bool:
    """Whether `op` SHOULD be hoisted (profitability, given legality).

    Phase A defaults:
      - legal AffineOp -> hoist (this is always profitable: saves
        trip_count-1 matmul+bias computations per inference)
      - ConvertOp -> NOT independently hoisted; only demand-driven
        when required by a hoisted AffineOp's dependency chain

    A future phase could add a real cost model here.
    """
    if not is_hoistable(op, analysis):
        return False

    # Only AffineOps are independently considered for hoisting.
    # ConvertOps are demand-driven (hoisted only when they feed
    # a hoisted AffineOp).
    return isinstance(op, AffineOp)


@dataclass
class HoistingStatistics:
    invariant_buffers: int = 0
    varying_buffers: int = 0
    affines_considered: int = 0
    affines_hoisted: int = 0
    converts_hoisted: int = 0


def _loop_producers(loop: TemporalLoop) -> Dict[str, ExecutionOp]:
    """Map buffer_name -> producing op, for ops in the loop body only."""
    result: Dict[str, ExecutionOp] = {}
    for op in loop.body:
        for out_name in op_outputs(op):
            result[out_name] = op
    return result


def _collect_demand_chain(
    target_op: ExecutionOp,
    loop_producers: Dict[str, ExecutionOp],
    analysis: TemporalInvarianceAnalysis,
    to_hoist: Set[str],
) -> None:
    """Demand-driven: collect `target_op` and any ConvertOps in its
    backward dependency chain that must also be hoisted to satisfy
    the target's data dependencies.

    Only traces through ConvertOps (the only pure-but-not-independently-
    hoisted op type in Phase A). Does NOT trace through AffineOps
    (which are independently identified for hoisting).
    """
    to_hoist.add(target_op.id)

    for input_name in op_runtime_inputs(target_op):
        if input_name in loop_producers:
            producer = loop_producers[input_name]
            if (isinstance(producer, ConvertOp)
                    and producer.id not in to_hoist
                    and is_hoistable(producer, analysis)):
                _collect_demand_chain(producer, loop_producers, analysis, to_hoist)


class InvariantAffineHoistingPass(CompilerPass):
    """B4 Phase A: hoists temporally invariant AffineOps (and their
    demand-driven ConvertOp dependencies) from the TemporalLoop into
    the module's preheader.

    This is a genuine Execution IR transformation: hoisted ops
    structurally leave the temporal loop. No `hoisted=True` metadata
    flag, no runtime predicate.

    Example (MNIST-like, static input):

    BEFORE::

        TemporalLoop x10:
            convert %input -> %stage0
            affine  %stage0 -> %current0
            lif     %current0, %mem0 -> %spike0
            ...

    AFTER::

        Preheader:
            convert %input -> %stage0
            affine  %stage0 -> %current0

        TemporalLoop x10:
            lif     %current0, %mem0 -> %spike0
            ...

    For SHD-like (temporal input), the pass does nothing (all affines
    are varying).
    """

    name = "InvariantAffineHoisting"

    def __init__(self):
        self.stats = HoistingStatistics()

    def run(self, module: ExecutionModule) -> ExecutionModule:
        analysis = TemporalInvarianceAnalysis.compute(module)

        self.stats = HoistingStatistics(
            invariant_buffers=len(analysis.invariant_buffers),
            varying_buffers=len(analysis.varying_buffers),
        )

        if not module.loop:
            return module

        loop = module.loop[0]

        # Phase 1: identify AffineOps that should be hoisted.
        hoistable_affine_ids: List[str] = []
        for op in loop.body:
            if isinstance(op, AffineOp):
                self.stats.affines_considered += 1
                if should_hoist(op, analysis):
                    hoistable_affine_ids.append(op.id)

        if not hoistable_affine_ids:
            return module

        # Phase 2: demand-driven dependency collection.
        # For each hoistable AffineOp, trace backward through its
        # ConvertOp dependency chain in the loop body.
        loop_prods = _loop_producers(loop)
        to_hoist_ids: Set[str] = set()

        for op in loop.body:
            if op.id in hoistable_affine_ids:
                _collect_demand_chain(op, loop_prods, analysis, to_hoist_ids)

        # Phase 3: split loop body into preheader + remaining,
        # preserving original program order (deterministic/topological).
        new_preheader_ops: List[ExecutionOp] = []
        remaining_ops: List[ExecutionOp] = []

        for op in loop.body:
            if op.id in to_hoist_ids:
                new_preheader_ops.append(op)
            else:
                remaining_ops.append(op)

        # Update statistics.
        for op in new_preheader_ops:
            if isinstance(op, AffineOp):
                self.stats.affines_hoisted += 1
            elif isinstance(op, ConvertOp):
                self.stats.converts_hoisted += 1

        # Build new module -- analysis is invalidated by construction
        # (fresh module; any subsequent pass/analysis must recompute).
        new_loop = dataclasses.replace(loop, body=tuple(remaining_ops))
        return dataclasses.replace(
            module,
            preheader_ops=tuple(module.preheader_ops) + tuple(new_preheader_ops),
            loop=(new_loop,),
        )
