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
    DelayPushOp,
    ExecutionModule,
    LIFExecutionSpec,
    LIFUpdateOp,
    StorageKind,
    TemporalLoop,
    TemporalMode,
)


from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel


def lower_source_ir(
    source: SNNModel,
    *,
    num_timesteps: Optional[int] = None,
    input_temporal_mode: TemporalMode = TemporalMode.VARYING,
) -> ExecutionModule:
    """Lowers `source` into an `ExecutionModule`.

    Args:
        source: A `[Linear -> Leaky]^N` source IR value. Should already
            have passed `spikeduet.ir.verifier.verify_module_ir` (the
            frontend calls it automatically; this function does not
            re-check source-level legality itself -- see this module's
            docstring).
        num_timesteps: How many times the model's timestep loop executes.
            Defaults to `source.num_steps` if the source IR already
            carries one (see `compiler/ir/model.py::SNNModel.num_steps`);
            as of B2 no frontend ever sets that field (it is a
            runtime/deployment concern the Python compiler deliberately
            does not own -- CLAUDE.md section 2), so callers almost always
            need to pass this explicitly. Raises `ValueError` if neither
            is available.
        input_temporal_mode: Whether the model's EXTERNAL input is the
            same vector on every timestep (`TemporalMode.INVARIANT` --
            e.g. MNIST: one image held constant) or genuinely different
            per timestep (`TemporalMode.VARYING` -- e.g. SHD). This is a
            deployment fact the source IR has no representation for and
            lowering cannot infer; see `compiler/ir/execution.py::
            TemporalMode`'s docstring. Defaults to `VARYING`, the
            conservative choice (claims no invariance a caller did not
            explicitly assert).

    Returns:
        An unverified `ExecutionModule` -- call `verify_execution_ir`
        separately, or use `spikeduet.ir.pass_manager.compile_source_to_execution_ir`
        for the combined lower-then-verify pipeline.

    Raises:
        ValueError: if `num_timesteps` is not given and `source.num_steps`
            is also unset, or if `num_timesteps` resolves to a
            non-positive value.
    """
    resolved_steps = num_timesteps if num_timesteps is not None else source.num_steps
    if resolved_steps is None:
        raise ValueError(
            "lower_source_ir: no num_timesteps given and source.num_steps is "
            "unset -- the timestep count is a runtime/deployment fact the "
            "source IR does not carry by default (see CLAUDE.md section 2); "
            "pass num_timesteps explicitly"
        )
    if resolved_steps <= 0:
        raise ValueError(f"lower_source_ir: num_timesteps must be positive, got {resolved_steps}")

    linear_layers: List[LinearLayer] = [l for l in source.layers if isinstance(l, LinearLayer)]
    lif_layers: List[LIFLayer] = [l for l in source.layers if isinstance(l, LIFLayer)]
    if not linear_layers or len(linear_layers) != len(lif_layers):
        raise ValueError(
            "lower_source_ir expects a valid [Linear -> Leaky]^N source IR "
            f"(got {len(linear_layers)} Linear and {len(lif_layers)} Leaky "
            "layers) -- run spikeduet.ir.verifier.verify_module_ir(source) first"
        )

    buffers: List[Buffer] = []
    ops: List = []

    input_name = "input"
    buffers.append(
        Buffer(
            name=input_name,
            shape=(linear_layers[0].in_features,),
            dtype=StorageKind.F32,
            kind=BufferKind.INPUT,
            stateful=False,
            temporal=input_temporal_mode,
        )
    )

    if source.readout_weights is not None and len(source.readout_weights) != resolved_steps:
        raise ValueError(
            f"lower_source_ir: readout_weights has {len(source.readout_weights)} entries "
            f"but the model runs {resolved_steps} timesteps"
        )

    prev_output = input_name
    for i, (linear, lif) in enumerate(zip(linear_layers, lif_layers)):
        stage_name = f"stage{i}"
        current_name = f"current{i}"
        mem_name = f"mem{i}"
        spike_name = f"spike{i}"

        # Every AffineOp is fed through an explicit ConvertOp -- from the
        # raw model input for layer 0, from the previous layer's spike
        # buffer otherwise -- see ConvertOp's docstring for why this is
        # emitted unconditionally, not only for the MVP-relevant case.
        buffers.append(
            Buffer(
                name=stage_name,
                shape=(linear.in_features,),
                dtype=StorageKind.F32,
                kind=BufferKind.STAGING,
                stateful=False,
                temporal=TemporalMode.VARYING,
            )
        )
        ops.append(ConvertOp(id=f"convert{i}", input=prev_output, output=stage_name))

        # FilterBank extension: a layer with per-synapse delays reads from a
        # stateful input-history ring that a DelayPushOp refreshes every
        # timestep, not directly from the staged vector.
        affine_input = stage_name
        if linear.synapse_delays is not None:
            depth = int(linear.synapse_delays.max())
            hist_name = f"hist{i}"
            buffers.append(
                Buffer(
                    name=hist_name,
                    shape=((depth + 1) * linear.in_features,),
                    dtype=StorageKind.F32,
                    kind=BufferKind.HISTORY,
                    stateful=True,
                    temporal=TemporalMode.VARYING,
                )
            )
            ops.append(DelayPushOp(id=f"delay{i}", input=stage_name, history=hist_name, depth=depth))
            affine_input = hist_name

        buffers.append(
            Buffer(
                name=current_name,
                shape=(linear.out_features,),
                dtype=StorageKind.F32,
                kind=BufferKind.CURRENT,
                stateful=False,
                temporal=TemporalMode.VARYING,
            )
        )
        ops.append(AffineOp(id=f"affine{i}", input=affine_input, weight_layer=linear, output=current_name))

        buffers.append(
            Buffer(
                name=mem_name,
                shape=(linear.out_features,),
                dtype=StorageKind.F32,
                kind=BufferKind.MEMBRANE,
                stateful=True,
                temporal=TemporalMode.VARYING,  # recurrent state, never invariant -- see Buffer.temporal
            )
        )
        buffers.append(
            Buffer(
                name=spike_name,
                shape=(linear.out_features,),
                dtype=StorageKind.F32,
                kind=BufferKind.SPIKE,
                stateful=False,
                temporal=TemporalMode.VARYING,
            )
        )
        spike_state_name = adapt_name = None
        if lif.spike_state:
            spike_state_name = f"sprev{i}"
            buffers.append(
                Buffer(
                    name=spike_state_name, shape=(linear.out_features,), dtype=StorageKind.F32,
                    kind=BufferKind.SPIKE_STATE, stateful=True, temporal=TemporalMode.VARYING,
                )
            )
        if lif.adaptation is not None:
            adapt_name = f"adapt{i}"
            buffers.append(
                Buffer(
                    name=adapt_name, shape=(linear.out_features,), dtype=StorageKind.F32,
                    kind=BufferKind.ADAPTATION, stateful=True, temporal=TemporalMode.VARYING,
                )
            )
        ops.append(
            LIFUpdateOp(
                id=f"lif{i}",
                current=current_name,
                membrane=mem_name,
                output_spike=spike_name,
                semantics=LIFExecutionSpec.from_source(lif, num_neurons=linear.out_features),
                spike_state=spike_state_name,
                adapt=adapt_name,
            )
        )

        prev_output = spike_name

    loop = TemporalLoop(id="t_loop", trip_count=resolved_steps, body=tuple(ops))

    return ExecutionModule(
        buffers=tuple(buffers),
        loop=(loop,),
        num_timesteps=resolved_steps,
        input_buffer=input_name,
        output_buffer=prev_output,
        preheader_ops=(),
        readout_weights=source.readout_weights,
    )


def _buffer(name: str, size: int, kind: BufferKind, *, stateful: bool = False,
            temporal: TemporalMode = TemporalMode.VARYING) -> Buffer:
    return Buffer(name=name, shape=(size,), dtype=StorageKind.F32, kind=kind, stateful=stateful,
                  temporal=temporal)
