"""Tests for spikeduet.ir.pass_manager -- CompilerPass/PassManager
infrastructure and the compile_source_to_execution_ir orchestration
function.
"""

import pytest

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import ExecutionModule
from spikeduet.ir.execution_verifier import ExecutionIRVerificationError
from spikeduet.ir.pass_manager import (
    CompilerPass,
    PassManager,
    VerifyExecutionIRPass,
    compile_source_to_execution_ir,
    default_pipeline,
)
from tests.test_reference import ChainSNN


def _source_ir():
    model = ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    return parse_snntorch_model(model)


def test_compile_source_to_execution_ir_returns_verified_module():
    ir = _source_ir()
    module = compile_source_to_execution_ir(ir, num_timesteps=5)

    assert isinstance(module, ExecutionModule)
    assert module.num_timesteps == 5


def test_default_pipeline_runs_verification_and_raises_on_bad_input():
    ir = _source_ir()
    # 0 timesteps is rejected by lowering itself (before the pipeline even
    # runs) -- use the pipeline directly against a module we've broken
    # post-lowering to prove the VerifyExecutionIRPass in the DEFAULT
    # pipeline is actually what's catching problems, not just lowering.
    import dataclasses

    module = compile_source_to_execution_ir(ir, num_timesteps=3)
    broken = dataclasses.replace(module, output_buffer="mem0")

    with pytest.raises(ExecutionIRVerificationError):
        default_pipeline().run(broken)


def test_passes_run_in_deterministic_order():
    order = []

    class RecordingPass(CompilerPass):
        def __init__(self, tag):
            self.tag = tag

        def run(self, module):
            order.append(self.tag)
            return module

    manager = PassManager([RecordingPass("a"), RecordingPass("b"), RecordingPass("c")])
    module = compile_source_to_execution_ir(_source_ir(), num_timesteps=1, pipeline=PassManager([]))

    manager.run(module)

    assert order == ["a", "b", "c"]


def test_verification_can_run_before_and_after_a_pass():
    from spikeduet.ir.lowering import lower_source_ir

    module = lower_source_ir(_source_ir(), num_timesteps=2)

    class IdentityPass(CompilerPass):
        def run(self, module):
            return module

    manager = PassManager([VerifyExecutionIRPass(), IdentityPass(), VerifyExecutionIRPass()])
    result = manager.run(module)

    assert result is module  # identity + verification-only passes never rebuild the module


def test_pass_failure_surfaces_as_plain_exception():
    class BoomPass(CompilerPass):
        def run(self, module):
            raise RuntimeError("boom")

    module = compile_source_to_execution_ir(_source_ir(), num_timesteps=1)
    manager = PassManager([BoomPass()])

    with pytest.raises(RuntimeError, match="boom"):
        manager.run(module)


def test_custom_pipeline_overrides_default():
    ir = _source_ir()
    calls = []

    class TrackingVerify(CompilerPass):
        def run(self, module):
            calls.append("custom")
            return module

    module = compile_source_to_execution_ir(ir, num_timesteps=1, pipeline=PassManager([TrackingVerify()]))

    assert calls == ["custom"]
    assert isinstance(module, ExecutionModule)
