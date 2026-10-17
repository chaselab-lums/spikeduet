"""Pass infrastructure and the standard pipelines.

Passes are ``ExecutionModule -> ExecutionModule`` functions; the IR is immutable, so a rewrite
returns a new module. A ``PassManager`` runs a fixed list of passes in order.

Pipelines (pick one by name with ``get_pipeline``):

=============  =====================================================
``baseline``   verify only: the naive schedule
``licm``       hoist timestep-invariant affines into the preheader
``fusion``     let each LIF write the next layer's staging buffer
``optimized``  LICM followed by fusion
=============  =====================================================
"""

from typing import Callable, Dict, List, Sequence

from spikeduet.ir.execution import ExecutionModule
from spikeduet.ir.execution_verifier import verify_execution_ir
from spikeduet.ir.lowering import lower_source_ir
from spikeduet.ir.model import SNNModel


class CompilerPass:
    """Base class for a pipeline stage."""

    name: str = "CompilerPass"

    def run(self, module: ExecutionModule) -> ExecutionModule:
        raise NotImplementedError


class VerifyExecutionIRPass(CompilerPass):
    """Runs ``verify_execution_ir`` and returns the module unchanged."""

    name = "VerifyExecutionIR"

    def run(self, module: ExecutionModule) -> ExecutionModule:
        verify_execution_ir(module)
        return module


class PassManager:
    """Runs ``passes`` front to back."""

    def __init__(self, passes: Sequence[CompilerPass]):
        self.passes: List[CompilerPass] = list(passes)

    def run(self, module: ExecutionModule) -> ExecutionModule:
        for p in self.passes:
            module = p.run(module)
        return module


def _verified(*passes: CompilerPass) -> PassManager:
    """Interleaves verification before, between and after ``passes``."""
    stages: List[CompilerPass] = [VerifyExecutionIRPass()]
    for p in passes:
        stages += [p, VerifyExecutionIRPass()]
    return PassManager(stages)


# licm.py and fusion.py import CompilerPass from this module, so the pipelines import them lazily.

def baseline_pipeline() -> PassManager:
    """Verification only."""
    return _verified()


def licm_pipeline() -> PassManager:
    """Loop-invariant affine hoisting."""
    from spikeduet.ir.licm import InvariantAffineHoistingPass
    return _verified(InvariantAffineHoistingPass())


def fusion_pipeline() -> PassManager:
    """Spike-staging fusion."""
    from spikeduet.ir.fusion import SpikeStagingFusionPass
    return _verified(SpikeStagingFusionPass())


def optimized_pipeline() -> PassManager:
    """LICM, then spike-staging fusion."""
    from spikeduet.ir.fusion import SpikeStagingFusionPass
    from spikeduet.ir.licm import InvariantAffineHoistingPass
    return _verified(InvariantAffineHoistingPass(), SpikeStagingFusionPass())


default_pipeline = baseline_pipeline
temporal_licm_pipeline = licm_pipeline

PIPELINES: Dict[str, Callable[[], PassManager]] = {
    "baseline": baseline_pipeline,
    "licm": licm_pipeline,
    "fusion": fusion_pipeline,
    "optimized": optimized_pipeline,
}


def get_pipeline(name: str) -> PassManager:
    """Returns a fresh pipeline by name; raises ``ValueError`` for an unknown name."""
    try:
        return PIPELINES[name]()
    except KeyError:
        raise ValueError(f"unknown pipeline {name!r}; available: {sorted(PIPELINES)}") from None


def compile_source_to_execution_ir(
    source: SNNModel,
    *,
    num_timesteps=None,
    input_temporal_mode=None,
    pipeline: PassManager = None,
) -> ExecutionModule:
    """Lowers ``source`` and runs ``pipeline`` (``baseline_pipeline()`` by default).

    ``num_timesteps`` and ``input_temporal_mode`` are forwarded to ``lower_source_ir``.
    """
    kwargs = {"num_timesteps": num_timesteps}
    if input_temporal_mode is not None:
        kwargs["input_temporal_mode"] = input_temporal_mode
    module = lower_source_ir(source, **kwargs)
    return (pipeline if pipeline is not None else baseline_pipeline()).run(module)
