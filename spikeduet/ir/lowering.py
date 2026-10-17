"""Lowering: source IR (``SNNModel``) to Execution IR (``ExecutionModule``).

A direct structural translation with no optimization: each ``[Linear -> LIF]`` pair becomes a
``ConvertOp -> AffineOp -> LIFUpdateOp`` triple inside one temporal loop. Verify the result with
``verify_execution_ir``, or use ``spikeduet.ir.pass_manager.compile_source_to_execution_ir``.
"""

from typing import List, Optional

from spikeduet.ir.execution import (
    AffineOp,
    Buffer,
    BufferKind,
    ConvertOp,
    ExecutionModule,
    LIFExecutionSpec,
    LIFUpdateOp,
    StorageKind,
    TemporalLoop,
    TemporalMode,
)
from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel


def _buffer(name: str, size: int, kind: BufferKind, *, stateful: bool = False,
            temporal: TemporalMode = TemporalMode.VARYING) -> Buffer:
    return Buffer(name=name, shape=(size,), dtype=StorageKind.F32, kind=kind, stateful=stateful,
                  temporal=temporal)


def lower_source_ir(
    source: SNNModel,
    *,
    num_timesteps: Optional[int] = None,
    input_temporal_mode: TemporalMode = TemporalMode.VARYING,
) -> ExecutionModule:
    """Lowers a verified ``[Linear -> LIF]^N`` model.

    Args:
        source: source IR, already checked by ``verify_module_ir``.
        num_timesteps: trip count of the timestep loop; defaults to ``source.num_steps``.
        input_temporal_mode: whether the external input is the same vector on every step
            (``INVARIANT``, e.g. a static image) or changes per step (``VARYING``, the default).

    Raises:
        ValueError: if no positive timestep count is available, or the layers do not pair up.
    """
    steps = num_timesteps if num_timesteps is not None else source.num_steps
    if steps is None:
        raise ValueError(
            "lower_source_ir: no num_timesteps given and source.num_steps is "
            "unset -- pass num_timesteps explicitly"
        )
    if steps <= 0:
        raise ValueError(f"lower_source_ir: num_timesteps must be positive, got {steps}")

    linear_layers: List[LinearLayer] = [l for l in source.layers if isinstance(l, LinearLayer)]
    lif_layers: List[LIFLayer] = [l for l in source.layers if isinstance(l, LIFLayer)]
    if not linear_layers or len(linear_layers) != len(lif_layers):
        raise ValueError(
            "lower_source_ir expects a valid [Linear -> Leaky]^N source IR "
            f"(got {len(linear_layers)} Linear and {len(lif_layers)} Leaky "
            "layers) -- run spikeduet.ir.verifier.verify_module_ir(source) first"
        )

    buffers = [_buffer("input", linear_layers[0].in_features, BufferKind.INPUT, temporal=input_temporal_mode)]
    ops: List = []
    prev = "input"
    for i, (linear, lif) in enumerate(zip(linear_layers, lif_layers)):
        n = linear.out_features
        buffers += [
            _buffer(f"stage{i}", linear.in_features, BufferKind.STAGING),
            _buffer(f"current{i}", n, BufferKind.CURRENT),
            _buffer(f"mem{i}", n, BufferKind.MEMBRANE, stateful=True),
            _buffer(f"spike{i}", n, BufferKind.SPIKE),
        ]
        ops += [
            ConvertOp(id=f"convert{i}", input=prev, output=f"stage{i}"),
            AffineOp(id=f"affine{i}", input=f"stage{i}", weight_layer=linear, output=f"current{i}"),
            LIFUpdateOp(id=f"lif{i}", current=f"current{i}", membrane=f"mem{i}", output_spike=f"spike{i}",
                        semantics=LIFExecutionSpec.from_source(lif, num_neurons=n)),
        ]
        prev = f"spike{i}"

    return ExecutionModule(
        buffers=tuple(buffers),
        loop=(TemporalLoop(id="t_loop", trip_count=steps, body=tuple(ops)),),
        num_timesteps=steps,
        input_buffer="input",
        output_buffer=prev,
    )
