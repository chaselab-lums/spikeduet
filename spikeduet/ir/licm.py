"""Temporal loop-invariant code motion.

``InvariantAffineHoistingPass`` moves every timestep-invariant ``AffineOp`` (and the convert
ops it depends on) from the temporal loop into the module preheader, so it runs once per
inference instead of once per step. With a static input this hoists the first layer's affine;
with a time-varying input nothing is hoistable and the module is returned unchanged.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Set

from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    ExecutionModule,
    ExecutionOp,
    TemporalLoop,
    op_is_pure,
    op_outputs,
    op_runtime_inputs,
)
from spikeduet.ir.execution_analysis import TemporalInvarianceAnalysis
from spikeduet.ir.pass_manager import CompilerPass


def is_hoistable(op: ExecutionOp, analysis: TemporalInvarianceAnalysis) -> bool:
    """Legality: ``op`` is pure and every runtime input is timestep-invariant."""
    return op_is_pure(op) and all(analysis.is_invariant(name) for name in op_runtime_inputs(op))


def should_hoist(op: ExecutionOp, analysis: TemporalInvarianceAnalysis) -> bool:
    """Profitability: hoist legal affines. Convert ops are only hoisted when an affine needs them."""
    return isinstance(op, AffineOp) and is_hoistable(op, analysis)


@dataclass
class HoistingStatistics:
    invariant_buffers: int = 0
    varying_buffers: int = 0
    affines_considered: int = 0
    affines_hoisted: int = 0
    converts_hoisted: int = 0


def _loop_producers(loop: TemporalLoop) -> Dict[str, ExecutionOp]:
    return {out: op for op in loop.body for out in op_outputs(op)}


def _collect_demand_chain(target_op: ExecutionOp, loop_producers: Dict[str, ExecutionOp],
                          analysis: TemporalInvarianceAnalysis, to_hoist: Set[str]) -> None:
    """Adds ``target_op`` and the hoistable convert ops feeding it to ``to_hoist``."""
    to_hoist.add(target_op.id)
    for name in op_runtime_inputs(target_op):
        producer = loop_producers.get(name)
        if isinstance(producer, ConvertOp) and producer.id not in to_hoist and is_hoistable(producer, analysis):
            _collect_demand_chain(producer, loop_producers, analysis, to_hoist)


class InvariantAffineHoistingPass(CompilerPass):
    """Moves invariant affines and their convert dependencies into ``preheader_ops``.

    Before::

        TemporalLoop x10:
            convert %input -> %stage0
            affine  %stage0 -> %current0
            lif     %current0, %mem0 -> %spike0

    After::

        Preheader:
            convert %input -> %stage0
            affine  %stage0 -> %current0
        TemporalLoop x10:
            lif     %current0, %mem0 -> %spike0
    """

    name = "InvariantAffineHoisting"

    def __init__(self):
        self.stats = HoistingStatistics()

    def run(self, module: ExecutionModule) -> ExecutionModule:
        analysis = TemporalInvarianceAnalysis.compute(module)
        self.stats = HoistingStatistics(invariant_buffers=len(analysis.invariant_buffers),
                                        varying_buffers=len(analysis.varying_buffers))
        if not module.loop:
            return module
        loop = module.loop[0]

        affines = [op for op in loop.body if isinstance(op, AffineOp)]
        self.stats.affines_considered = len(affines)
        targets = [op for op in affines if should_hoist(op, analysis)]
        if not targets:
            return module

        producers = _loop_producers(loop)
        to_hoist: Set[str] = set()
        for op in targets:
            _collect_demand_chain(op, producers, analysis, to_hoist)

        # Program order is preserved in both halves.
        hoisted: List[ExecutionOp] = [op for op in loop.body if op.id in to_hoist]
        remaining = [op for op in loop.body if op.id not in to_hoist]
        self.stats.affines_hoisted = sum(isinstance(op, AffineOp) for op in hoisted)
        self.stats.converts_hoisted = sum(isinstance(op, ConvertOp) for op in hoisted)

        return dataclasses.replace(
            module,
            preheader_ops=tuple(module.preheader_ops) + tuple(hoisted),
            loop=(dataclasses.replace(loop, body=tuple(remaining)),),
        )
