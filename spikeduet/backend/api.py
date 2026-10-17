"""Compiler entry point: source IR -> pass pipeline -> backend.

```python
compile_model(source_ir, backend="portable_c", num_timesteps=10,
              input_temporal_mode=TemporalMode.INVARIANT, pipeline="optimized")
```

This is the only place a backend is selected by name; the IR never branches on a target.
"""

from typing import Optional, Union

from spikeduet.backend.base import Backend, GeneratedArtifact
from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.portable_c import PortableCBackend
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.model import SNNModel
from spikeduet.ir.pass_manager import PassManager, compile_source_to_execution_ir, get_pipeline
from spikeduet.ir.verifier import verify_module_ir

_BACKENDS = {
    "portable_c": PortableCBackend,
    "efr32": Efr32Backend,
}


def get_backend(name: str) -> Backend:
    """Instantiates a backend by name; raises ``ValueError`` listing the valid names."""
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available backends: {sorted(_BACKENDS)}"
        ) from None


def compile_model(
    source: SNNModel,
    *,
    backend: str,
    num_timesteps: Optional[int] = None,
    input_temporal_mode: TemporalMode = TemporalMode.VARYING,
    pipeline: Union[str, PassManager, None] = None,
    **backend_kwargs,
) -> GeneratedArtifact:
    """Verifies ``source``, lowers it, runs ``pipeline`` and generates code with ``backend``.

    Args:
        source: a parsed ``[Linear -> Leaky]^N`` model.
        backend: ``"portable_c"`` or ``"efr32"``.
        num_timesteps: trip count of the timestep loop.
        input_temporal_mode: ``TemporalMode.INVARIANT`` for an input held constant over all steps,
            ``TemporalMode.VARYING`` (default) for one input vector per step.
        pipeline: a pipeline name (``"baseline"``, ``"licm"``, ``"fusion"``, ``"optimized"``) or a
            ``PassManager``. Defaults to ``"baseline"``.
        **backend_kwargs: forwarded to the backend's ``generate()`` (see each backend).
    """
    verify_module_ir(source)
    if isinstance(pipeline, str):
        pipeline = get_pipeline(pipeline)
    module = compile_source_to_execution_ir(
        source, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode, pipeline=pipeline
    )
    return get_backend(backend).generate(module, **backend_kwargs)
