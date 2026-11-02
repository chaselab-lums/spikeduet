"""Structural verifier for the Execution IR.

Checks dataflow and schedule invariants only (references resolve, shapes agree, nothing is read
before it is defined, preheader legality). Hardware limits are a backend concern.
"""


from typing import Set


from spikeduet.ir.execution import (
    AffineOp,
    BufferKind,
    ConvertOp,
    DelayPushOp,
    ExecutionModule,
    LIFUpdateOp,
    TemporalLoop,
    TemporalMode,
)


class ExecutionIRVerificationError(Exception):
    """An ``ExecutionModule`` violates an Execution IR invariant."""


def _fail(detail: str) -> None:
    raise ExecutionIRVerificationError(f"Execution IR verification failed:\n{detail}")


def _affine_input_size(op: AffineOp) -> int:
    """Element count the affine's input buffer must have: `in_features`, or
    for a layer with per-synapse delays the whole `(D+1) * in_features`
    history ring it gathers from."""
    layer = op.weight_layer
    if layer.synapse_delays is None:
        return layer.in_features
    return (int(layer.synapse_delays.max()) + 1) * layer.in_features


def _shape_size(shape) -> int:
    size = 1
    for d in shape:
        size *= d
    return size


def verify_execution_ir(module: ExecutionModule) -> None:
    """Verifies that `module` is an internally consistent, well-formed
    Execution IR value.

    Checks:
      * every buffer name referenced by any op exists in `module.buffers`
      * `module.input_buffer`/`module.output_buffer` exist and are the
        module's actual INPUT/SPIKE-kind buffers
      * `AffineOp.input` buffer size matches `weight_layer.in_features`;
        `AffineOp.output` buffer size matches `weight_layer.out_features`
      * `LIFUpdateOp`'s current/membrane/output_spike buffers all have
        size `semantics.num_neurons`; the membrane buffer is `stateful`
      * `ConvertOp.input`/`.output` buffers agree in size (element count;
        dtype is deliberately NOT required to match -- a `ConvertOp` is
        exactly where a dtype change is allowed to happen, see
        `compiler/ir/execution.py::ConvertOp`)
      * every op id is unique module-wide (across preheader + loop)
      * preheader ops (B4 Phase A): no LIFUpdateOp, no consumption of
        a VARYING external input, valid def-use within the preheader,
        buffer shapes valid
      * within the loop body, no buffer is read before it is defined:
        the model input and every stateful (membrane) buffer are defined
        before the loop starts; every other buffer must be written by an
        earlier op in the SAME body before any later op reads it;
        preheader-produced buffers are also available in the loop body
      * the temporal region itself is structurally valid: positive
        `trip_count`, non-empty `body`

    Raises:
        ExecutionIRVerificationError: on the first invariant violated.
    """
    buffer_names = {b.name for b in module.buffers}

    if module.input_buffer not in buffer_names:
        _fail(f"module.input_buffer {module.input_buffer!r} is not a defined buffer")
    input_buf = module.get_buffer(module.input_buffer)
    if input_buf.kind is not BufferKind.INPUT:
        _fail(f"module.input_buffer {module.input_buffer!r} is not kind INPUT ({input_buf.kind})")

    if module.output_buffer not in buffer_names:
        _fail(f"module.output_buffer {module.output_buffer!r} is not a defined buffer")
    output_buf = module.get_buffer(module.output_buffer)
    if output_buf.kind is not BufferKind.SPIKE:
        _fail(f"module.output_buffer {module.output_buffer!r} is not kind SPIKE ({output_buf.kind})")

    if not module.loop:
        _fail("module has no top-level TemporalLoop region")

    if module.readout_weights is not None and len(module.readout_weights) != module.num_timesteps:
        _fail(
            f"readout_weights has {len(module.readout_weights)} entries "
            f"but the module runs {module.num_timesteps} timesteps"
        )

    seen_op_ids: Set[str] = set()

    # --- Preheader verification (B4 Phase A) ---
    # Buffers available before the preheader: model input + stateful
    # (membrane) buffers, zero-initialized before everything.
    preheader_defined: Set[str] = {module.input_buffer}
    preheader_defined |= {b.name for b in module.buffers if b.stateful}

    for op in module.preheader_ops:
        if op.id in seen_op_ids:
            _fail(f"duplicate op id {op.id!r}")
        seen_op_ids.add(op.id)

        # Stateful ops must remain in the temporal loop.
        if isinstance(op, LIFUpdateOp):
            _fail(
                f"LIFUpdateOp {op.id!r} is not allowed in preheader "
                "(stateful ops must remain in temporal loop)"
            )

        if isinstance(op, DelayPushOp):
            _fail(
                f"DelayPushOp {op.id!r} is not allowed in preheader "
                "(it updates a stateful history ring and must stay in the temporal loop)"
            )

        if isinstance(op, ConvertOp):
            # Reject consumption of a VARYING external input in preheader.
            if op.input == module.input_buffer and input_buf.temporal is TemporalMode.VARYING:
                _fail(
                    f"preheader op {op.id!r} consumes varying external input "
                    f"{op.input!r} -- preheader ops must only consume "
                    "temporally invariant inputs"
                )
            _verify_reads(op.id, [op.input], preheader_defined, buffer_names)
            in_buf = module.get_buffer(op.input)
            out_buf = _require_buffer(module, buffer_names, op.output, op.id, "output")
            if _shape_size(in_buf.shape) != _shape_size(out_buf.shape):
                _fail(
                    f"ConvertOp {op.id!r} input {op.input!r} has size "
                    f"{_shape_size(in_buf.shape)} but output {op.output!r} "
                    f"has size {_shape_size(out_buf.shape)}"
                )
            preheader_defined.add(op.output)

        elif isinstance(op, AffineOp):
            # Reject consumption of a VARYING external input in preheader.
            if op.input == module.input_buffer and input_buf.temporal is TemporalMode.VARYING:
                _fail(
                    f"preheader op {op.id!r} consumes varying external input "
                    f"{op.input!r} -- preheader ops must only consume "
                    "temporally invariant inputs"
                )
            _verify_reads(op.id, [op.input], preheader_defined, buffer_names)
            in_buf = module.get_buffer(op.input)
            out_buf = _require_buffer(module, buffer_names, op.output, op.id, "output")
            in_features = _affine_input_size(op)
            out_features = op.weight_layer.out_features
            if _shape_size(in_buf.shape) != in_features:
                _fail(
                    f"AffineOp {op.id!r} input {op.input!r} has size "
                    f"{_shape_size(in_buf.shape)}\nbut weight_layer.in_features "
                    f"= {in_features}"
                )
            if _shape_size(out_buf.shape) != out_features:
                _fail(
                    f"AffineOp {op.id!r} output {op.output!r} has size "
                    f"{_shape_size(out_buf.shape)}\nbut weight_layer.out_features "
                    f"= {out_features}"
                )
            preheader_defined.add(op.output)

        else:
            _fail(f"unrecognized op type {type(op).__name__} (id={getattr(op, 'id', '?')!r})")

    # --- Temporal-loop verification ---
    for region in module.loop:
        if not isinstance(region, TemporalLoop):
            _fail(f"top-level region {region!r} is not a TemporalLoop")
        if region.trip_count <= 0:
            _fail(f"TemporalLoop {region.id!r} has non-positive trip_count {region.trip_count}")
        if not region.body:
            _fail(f"TemporalLoop {region.id!r} has an empty body")

        # Buffers "defined" before the loop body begins executing: the
        # model input (supplied by the caller each inference) and every
        # stateful (membrane) buffer (zero-initialized once, before the
        # loop -- see compiler/ir/execution.py::Buffer.stateful), PLUS
        # every buffer defined by preheader ops (B4 Phase A).
        defined: Set[str] = set(preheader_defined)

        for op in region.body:
            if op.id in seen_op_ids:
                _fail(f"duplicate op id {op.id!r}")
            seen_op_ids.add(op.id)

            if isinstance(op, ConvertOp):
                _verify_reads(op.id, [op.input], defined, buffer_names)
                in_buf = module.get_buffer(op.input)
                out_buf = _require_buffer(module, buffer_names, op.output, op.id, "output")
                if _shape_size(in_buf.shape) != _shape_size(out_buf.shape):
                    _fail(
                        f"ConvertOp {op.id!r} input {op.input!r} has size "
                        f"{_shape_size(in_buf.shape)} but output {op.output!r} "
                        f"has size {_shape_size(out_buf.shape)}"
                    )
                defined.add(op.output)

            elif isinstance(op, AffineOp):
                _verify_reads(op.id, [op.input], defined, buffer_names)
                in_buf = module.get_buffer(op.input)
                out_buf = _require_buffer(module, buffer_names, op.output, op.id, "output")
                in_features = _affine_input_size(op)
                out_features = op.weight_layer.out_features
                if _shape_size(in_buf.shape) != in_features:
                    _fail(
                        f"AffineOp {op.id!r} input {op.input!r} has size "
                        f"{_shape_size(in_buf.shape)}\nbut weight_layer.in_features "
                        f"= {in_features}"
                    )
                if _shape_size(out_buf.shape) != out_features:
                    _fail(
                        f"AffineOp {op.id!r} output {op.output!r} has size "
                        f"{_shape_size(out_buf.shape)}\nbut weight_layer.out_features "
                        f"= {out_features}"
                    )
                defined.add(op.output)

            elif isinstance(op, DelayPushOp):
                _verify_reads(op.id, [op.input, op.history], defined, buffer_names)
                in_buf = module.get_buffer(op.input)
                hist_buf = module.get_buffer(op.history)
                if not hist_buf.stateful:
                    _fail(f"DelayPushOp {op.id!r} history buffer {op.history!r} is not marked stateful")
                if op.depth < 0 or _shape_size(hist_buf.shape) != (op.depth + 1) * _shape_size(in_buf.shape):
                    _fail(
                        f"DelayPushOp {op.id!r} history {op.history!r} has size {_shape_size(hist_buf.shape)} "
                        f"but expected (depth+1) * input size = {(op.depth + 1) * _shape_size(in_buf.shape)}"
                    )
                defined.add(op.history)

            elif isinstance(op, LIFUpdateOp):
                _verify_reads(op.id, [op.current, op.membrane], defined, buffer_names)
                for flag, buf_name, label in (
                    (op.semantics.spike_state, op.spike_state, "spike_state"),
                    (op.semantics.adaptation is not None, op.adapt, "adapt"),
                ):
                    if flag != (buf_name is not None):
                        _fail(
                            f"LIFUpdateOp {op.id!r}: semantics require a {label} buffer = {flag} "
                            f"but the op's {label} field is {buf_name!r}"
                        )
                    if buf_name is not None:
                        _verify_reads(op.id, [buf_name], defined, buffer_names)
                        extra = module.get_buffer(buf_name)
                        if not extra.stateful or _shape_size(extra.shape) != op.semantics.num_neurons:
                            _fail(
                                f"LIFUpdateOp {op.id!r} {label} buffer {buf_name!r} must be stateful "
                                f"with size {op.semantics.num_neurons}"
                            )
                n = op.semantics.num_neurons
                current_buf = module.get_buffer(op.current)
                membrane_buf = _require_buffer(module, buffer_names, op.membrane, op.id, "membrane")
                spike_buf = _require_buffer(module, buffer_names, op.output_spike, op.id, "output_spike")
                if _shape_size(current_buf.shape) != n:
                    _fail(
                        f"LIFUpdateOp {op.id!r} current {op.current!r} has size "
                        f"{_shape_size(current_buf.shape)}\nbut semantics.num_neurons = {n}"
                    )
                if _shape_size(membrane_buf.shape) != n:
                    _fail(
                        f"LIFUpdateOp {op.id!r} membrane {op.membrane!r} has size "
                        f"{_shape_size(membrane_buf.shape)}\nbut semantics.num_neurons = {n}"
                    )
                if not membrane_buf.stateful:
                    _fail(
                        f"LIFUpdateOp {op.id!r} membrane buffer {op.membrane!r} is "
                        "not marked stateful"
                    )
                if _shape_size(spike_buf.shape) != n:
                    _fail(
                        f"LIFUpdateOp {op.id!r} output_spike {op.output_spike!r} has "
                        f"size {_shape_size(spike_buf.shape)}\nbut semantics.num_neurons = {n}"
                    )
                defined.add(op.output_spike)
                # membrane was already in `defined` (stateful, pre-loop) and
                # remains so -- this op updates it in place, not a fresh definition.

            else:
                _fail(f"unrecognized op type {type(op).__name__} (id={getattr(op, 'id', '?')!r})")


def _require_buffer(module: ExecutionModule, buffer_names: Set[str], name: str, op_id: str, role: str):
    if name not in buffer_names:
        _fail(f"op {op_id!r} references undefined buffer {name!r} (as {role})")
    return module.get_buffer(name)


def _verify_reads(op_id: str, read_names, defined: Set[str], buffer_names: Set[str]) -> None:
    for name in read_names:
        if name not in buffer_names:
            _fail(f"op {op_id!r} references undefined buffer {name!r}")
        if name not in defined:
            _fail(
                f"op {op_id!r} consumes buffer {name!r} before it is defined "
                "(no earlier op in this TemporalLoop body produced it, and it "
                "is neither the model input nor a stateful buffer)"
            )


def _size(module: ExecutionModule, name: str) -> int:
    size = 1
    for d in module.get_buffer(name).shape:
        size *= d
    return size


def _verify_op(module: ExecutionModule, op, defined: Set[str], names: Set[str], seen_ids: Set[str]) -> None:
    """Checks one op against the buffers defined so far and records what it defines."""
    if op.id in seen_ids:
        _fail(f"duplicate op id {op.id!r}")
    seen_ids.add(op.id)

    if isinstance(op, ConvertOp):
        _verify_reads(op.id, [op.input], defined, names)
        _require_buffer(names, op.output, op.id, "output")
        if _size(module, op.input) != _size(module, op.output):
            _fail(
                f"ConvertOp {op.id!r} input {op.input!r} has size {_size(module, op.input)} "
                f"but output {op.output!r} has size {_size(module, op.output)}"
            )
        defined.add(op.output)
    elif isinstance(op, AffineOp):
        _verify_reads(op.id, [op.input], defined, names)
        _require_buffer(names, op.output, op.id, "output")
        w = op.weight_layer
        if _size(module, op.input) != w.in_features:
            _fail(
                f"AffineOp {op.id!r} input {op.input!r} has size {_size(module, op.input)}\n"
                f"but weight_layer.in_features = {w.in_features}"
            )
        if _size(module, op.output) != w.out_features:
            _fail(
                f"AffineOp {op.id!r} output {op.output!r} has size {_size(module, op.output)}\n"
                f"but weight_layer.out_features = {w.out_features}"
            )
        defined.add(op.output)
    elif isinstance(op, LIFUpdateOp):
        _verify_reads(op.id, [op.current, op.membrane], defined, names)
        _require_buffer(names, op.output_spike, op.id, "output_spike")
        n = op.semantics.num_neurons
        for role, name in (("current", op.current), ("membrane", op.membrane), ("output_spike", op.output_spike)):
            if _size(module, name) != n:
                _fail(
                    f"LIFUpdateOp {op.id!r} {role} {name!r} has size {_size(module, name)}\n"
                    f"but semantics.num_neurons = {n}"
                )
        if not module.get_buffer(op.membrane).stateful:
            _fail(f"LIFUpdateOp {op.id!r} membrane buffer {op.membrane!r} is not marked stateful")
        defined.add(op.output_spike)
    else:
        _fail(f"unrecognized op type {type(op).__name__} (id={getattr(op, 'id', '?')!r})")
