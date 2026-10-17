"""Convenience entry point: source IR -> ``model_data.h``.

The header is vendor neutral and identical for every backend: dimension macros, the
``iscas_layer_t`` descriptor type, one flattened ``static const float`` array per weight, bias
and LIF parameter, and the ``ISCAS_LAYERS[ISCAS_NUM_LAYERS]`` descriptor table. A generic
runtime loops over that table, so no per-model C source is needed. The timestep count and the
readout are not part of the header.
"""

from spikeduet.backend.c_common import render_model_data_header
from spikeduet.ir.model import SNNModel
from spikeduet.ir.pass_manager import compile_source_to_execution_ir
from spikeduet.ir.verifier import verify_module_ir


def generate_c_model(ir: SNNModel) -> str:
    """Returns the text of ``model_data.h`` for ``ir``; deterministic for a given model."""
    verify_module_ir(ir)
    # The header does not depend on the timestep count; any valid value lowers the same layers.
    module = compile_source_to_execution_ir(ir, num_timesteps=ir.num_steps or 1)
    return render_model_data_header(module)


def write_c_model(ir: SNNModel, path: str) -> None:
    """Writes ``generate_c_model(ir)`` to ``path``."""
    with open(path, "w") as f:
        f.write(generate_c_model(ir))
