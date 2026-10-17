"""Spike-staging fusion.

Every layer boundary of the lowered IR is::

    LIFUpdateOp(output_spike=%spikeN) -> ConvertOp(%spikeN -> %stageM) -> AffineOp(input=%stageM)

The convert op is an identity staging copy, so ``SpikeStagingFusionPass`` deletes it and lets the
LIF write its final spike straight into ``%stageM``. A LIF's ``output_spike`` is already the
fully resolved value (quantization, inhibition, reset and grading happen inside the LIF update),
so only the destination of that value changes, never the value itself.

A LIF is fused when its spike buffer has exactly one consumer, that consumer is a convert op,
and the convert's output has exactly one consumer, an affine op. The model output has no
consumer and is therefore never fused. Spike buffers left without readers are dropped. Fusion
only touches the loop body; LIF updates are never hoisted, so it composes with LICM.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    ExecutionModule,
    ExecutionOp,
    LIFUpdateOp,
)
from spikeduet.ir.execution_analysis import is_dead_after_removing_ops, single_consumer
from spikeduet.ir.pass_manager import CompilerPass


@dataclass
class FusionStatistics:
    candidates_considered: int = 0
    fusions_applied: int = 0
    convert_ops_removed: int = 0
    buffers_eliminated: int = 0


def _op_by_id(module: ExecutionModule, op_id: str) -> Optional[ExecutionOp]:
    for op in module.walk_ops():
        if op.id == op_id:
            return op
    return None


def find_fusion_candidate(
    module: ExecutionModule, lif_op: LIFUpdateOp
) -> Optional[Tuple[ConvertOp, AffineOp]]:
    """``(convert_op, affine_op)`` if ``lif_op`` can be fused, else ``None``."""
    spike_buf = lif_op.output_spike

    convert_id = single_consumer(module, spike_buf)
    if convert_id is None:
        return None
    convert_op = _op_by_id(module, convert_id)
    if not isinstance(convert_op, ConvertOp) or convert_op.input != spike_buf:
        return None

    stage_buf = convert_op.output
    affine_id = single_consumer(module, stage_buf)
    if affine_id is None:
        return None
    affine_op = _op_by_id(module, affine_id)
    if not isinstance(affine_op, AffineOp) or affine_op.input != stage_buf:
        return None

    return convert_op, affine_op


class SpikeStagingFusionPass(CompilerPass):
    """Rewrites fusable LIF ops to write the downstream staging buffer and removes the copies."""

    name = "SpikeStagingFusion"

    def __init__(self):
        self.stats = FusionStatistics()

    def run(self, module: ExecutionModule) -> ExecutionModule:
        self.stats = FusionStatistics()
        if not module.loop:
            return module
        loop = module.loop[0]

        # Candidates are independent, so decide them all on the original module.
        fusions: Dict[str, Tuple[ConvertOp, AffineOp]] = {}
        for op in loop.body:
            if isinstance(op, LIFUpdateOp):
                self.stats.candidates_considered += 1
                candidate = find_fusion_candidate(module, op)
                if candidate is not None:
                    fusions[op.id] = candidate

        if not fusions:
            return module

        removed_convert_ids: Set[str] = {convert_op.id for convert_op, _ in fusions.values()}

        new_body: List[ExecutionOp] = []
        for op in loop.body:
            if op.id in removed_convert_ids:
                continue
            if isinstance(op, LIFUpdateOp) and op.id in fusions:
                convert_op, _ = fusions[op.id]
                new_body.append(dataclasses.replace(op, output_spike=convert_op.output))
                self.stats.fusions_applied += 1
                self.stats.convert_ops_removed += 1
            else:
                new_body.append(op)

        # Drop the spike buffers whose only reader was a removed convert.
        dead_buffer_names: Set[str] = set()
        for convert_op, _ in fusions.values():
            spike_buf = convert_op.input
            if is_dead_after_removing_ops(module, spike_buf, removed_convert_ids):
                dead_buffer_names.add(spike_buf)

        new_buffers = tuple(b for b in module.buffers if b.name not in dead_buffer_names)
        self.stats.buffers_eliminated = len(module.buffers) - len(new_buffers)

        new_loop = dataclasses.replace(loop, body=tuple(new_body))
        return dataclasses.replace(module, loop=(new_loop,), buffers=new_buffers)
