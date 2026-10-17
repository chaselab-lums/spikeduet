"""Deterministic text dump of an ``ExecutionModule`` (buffers, preheader, loop body). Never prints weights."""

from spikeduet.ir.execution import (
    AffineOp,
    ConvertOp,
    ExecutionModule,
    LIFUpdateOp,
    TemporalMode,
)


def _buffer_line(module: ExecutionModule, name: str) -> str:
    b = module.get_buffer(name)
    size = b.shape[0] if b.shape else 0
    tag = "invariant" if b.temporal is TemporalMode.INVARIANT else "varying"
    role = "state" if b.stateful else b.kind.value
    return f"  %{name:<12}: {b.dtype.value}[{size}] {role} ({tag})"


def _format_op(op) -> str:
    if isinstance(op, ConvertOp):
        return f"  convert %{op.input} -> %{op.output}"
    if isinstance(op, AffineOp):
        return (
            f"  affine  %{op.input} -> %{op.output}  "
            f"({op.weight_layer.in_features} -> {op.weight_layer.out_features})"
        )
    if isinstance(op, LIFUpdateOp):
        s = op.semantics
        return (
            f"  lif     %{op.current}, %{op.membrane} -> %{op.output_spike}  "
            f"(reset={s.reset_mechanism}, reset_delay={s.reset_delay}, "
            f"inhibition={s.inhibition}, state_quant="
            f"{'set' if s.state_quant is not None else 'None'})"
        )
    return f"  <unrecognized op: {type(op).__name__}>"


def format_execution_ir(module: ExecutionModule) -> str:
    """Renders `module` as a concise, deterministic text dump.

    Example, after ``InvariantAffineHoistingPass`` on a static-input model::

        ExecutionModule timesteps=10

        Buffers:
          %input       : f32[784] input (invariant)
          %stage0      : f32[784] staging (varying)
          %current0    : f32[64] current (varying)
          ...

        Preheader:
          convert %input -> %stage0
          affine  %stage0 -> %current0  (784 -> 64)

        TemporalLoop t_loop x10:
          lif     %current0, %mem0 -> %spike0
          ...
    """
    lines = [f"ExecutionModule timesteps={module.num_timesteps}", ""]

    lines.append("Buffers:")
    for b in module.buffers:
        lines.append(_buffer_line(module, b.name))
    lines.append("")

    lines.append("Preheader:")
    if module.preheader_ops:
        for op in module.preheader_ops:
            lines.append(_format_op(op))
    else:
        lines.append("  <empty>")
    lines.append("")

    for region in module.loop:
        lines.append(f"TemporalLoop {region.id} x{region.trip_count}:")
        for op in region.body:
            lines.append(_format_op(op))

    return "\n".join(lines)
