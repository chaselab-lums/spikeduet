"""SpikeDuet: compile stateful snnTorch spiking networks to C for microcontrollers.

Typical use::

    from spikeduet import parse_snntorch_model, compile_model, TemporalMode

    ir = parse_snntorch_model(model)
    artifact = compile_model(ir, backend="portable_c", num_timesteps=25,
                             input_temporal_mode=TemporalMode.VARYING, pipeline="optimized")
    artifact.write("build/")
"""

from spikeduet.backend.api import compile_model, get_backend
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, get_pipeline
from spikeduet.ir.verifier import verify_module_ir
from spikeduet.reference.simulator import run_sequence

__version__ = "0.1.0"

__all__ = [
    "TemporalMode",
    "compile_model",
    "compile_source_to_execution_ir",
    "get_backend",
    "get_pipeline",
    "parse_snntorch_model",
    "run_sequence",
    "verify_module_ir",
]
