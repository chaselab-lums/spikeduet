"""Structural verifier for the Execution IR.

Checks dataflow and schedule invariants only (references resolve, shapes agree, nothing is read
before it is defined, preheader legality). Hardware limits are a backend concern.
"""

from typing import Iterable, Set

from spikeduet.ir.execution import (
    AffineOp,
    BufferKind,
    ConvertOp,
    ExecutionModule,
    LIFUpdateOp,
    TemporalLoop,
    TemporalMode,
)


class ExecutionIRVerificationError(Exception):
    """An ``ExecutionModule`` violates an Execution IR invariant."""


def _fail(detail: str) -> None:
    raise ExecutionIRVerificationError(f"Execution IR verification failed:\n{detail}")


def _size(module: ExecutionModule, name: str) -> int:
    size = 1
    for d in module.get_buffer(name).shape:
        size *= d
    return size


def verify_execution_ir(module: ExecutionModule) -> None:
    """Verifies ``module``.

    Checks that every referenced buffer exists; the module input/output are INPUT/SPIKE
    buffers; affine, LIF and convert shapes agree; op ids are unique; the preheader holds no
    LIF update and does not read a varying external input; every buffer is defined before it is
    read (the input, the membranes and preheader outputs are available from the start of each
    iteration); and each loop has a positive trip count and a non-empty body.

    Raises:
        ExecutionIRVerificationError: on the first violation.
    """
    names = {b.name for b in module.buffers}
    for attr, kind in (("input_buffer", BufferKind.INPUT), ("output_buffer", BufferKind.SPIKE)):
        name = getattr(module, attr)
        if name not in names:
            _fail(f"module.{attr} {name!r} is not a defined buffer")
        buf = module.get_buffer(name)
        if buf.kind is not kind:
            _fail(f"module.{attr} {name!r} is not kind {kind.name} ({buf.kind})")
    if not module.loop:
        _fail("module has no top-level TemporalLoop region")

    varying_input = module.get_buffer(module.input_buffer).temporal is TemporalMode.VARYING
    seen_ids: Set[str] = set()
    defined: Set[str] = {module.input_buffer} | {b.name for b in module.buffers if b.stateful}

    for op in module.preheader_ops:
        if isinstance(op, LIFUpdateOp):
            _fail(f"LIFUpdateOp {op.id!r} is not allowed in preheader (stateful ops must remain in temporal loop)")
        if isinstance(op, (ConvertOp, AffineOp)) and op.input == module.input_buffer and varying_input:
            _fail(
                f"preheader op {op.id!r} consumes varying external input {op.input!r} -- "
                "preheader ops must only consume temporally invariant inputs"
            )
        _verify_op(module, op, defined, names, seen_ids)

    for region in module.loop:
        if not isinstance(region, TemporalLoop):
            _fail(f"top-level region {region!r} is not a TemporalLoop")
        if region.trip_count <= 0:
            _fail(f"TemporalLoop {region.id!r} has non-positive trip_count {region.trip_count}")
        if not region.body:
            _fail(f"TemporalLoop {region.id!r} has an empty body")
        body_defined = set(defined)
        for op in region.body:
            _verify_op(module, op, body_defined, names, seen_ids)


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


def _require_buffer(names: Set[str], name: str, op_id: str, role: str) -> None:
    if name not in names:
        _fail(f"op {op_id!r} references undefined buffer {name!r} (as {role})")


def _verify_reads(op_id: str, read_names: Iterable[str], defined: Set[str], names: Set[str]) -> None:
    for name in read_names:
        if name not in names:
            _fail(f"op {op_id!r} references undefined buffer {name!r}")
        if name not in defined:
            _fail(
                f"op {op_id!r} consumes buffer {name!r} before it is defined "
                "(no earlier op in this TemporalLoop body produced it, and it "
                "is neither the model input nor a stateful buffer)"
            )
