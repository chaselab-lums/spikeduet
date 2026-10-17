"""Target descriptions: memory capacities and the optional matrix accelerator.

Nothing in ``spikeduet.ir`` reads a ``PlatformSpec``; backends and the placement planner do.
Constraints that have not been established are ``None`` rather than guessed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AcceleratorSpec:
    """A matrix/vector accelerator.

    ``supported_ops`` and ``supported_dtypes`` are descriptive strings. ``matrix_limits`` and
    ``alignment_bytes`` are ``None`` when no limit has been established.
    """

    name: str
    supported_ops: Tuple[str, ...]
    supported_dtypes: Tuple[str, ...]
    matrix_limits: Optional[str] = None
    alignment_bytes: Optional[int] = None


@dataclass(frozen=True)
class PlatformSpec:
    """A deployment target.

    Attributes:
        flash_bytes, sram_bytes: capacities, or ``None`` if unknown. The placement planner
            checks its SRAM estimate against ``sram_bytes``.
        alignment_bytes: required CPU buffer alignment, or ``None``.
        supported_numeric_types: e.g. ``("float32", "float16")``.
        accelerator: the matrix accelerator, or ``None`` for a CPU-only target.
        allocation_policy: how runtime buffers are allocated (``"dynamic"`` = checked malloc in
            ``snn_init()``).
    """

    name: str
    flash_bytes: Optional[int]
    sram_bytes: Optional[int]
    alignment_bytes: Optional[int]
    supported_numeric_types: Tuple[str, ...]
    accelerator: Optional[AcceleratorSpec] = None
    allocation_policy: str = "dynamic"


# EFR32MG24B310F1536IM48 (xG24-DK2601B). Flash and SRAM are from the Silicon Labs datasheet.
# The MVP is used only for dense FP16 matrix-vector products; no dimension or alignment limit is
# enforced because none has been established.
EFR32MG24_MVP_ACCELERATOR = AcceleratorSpec(
    name="EFR32MG24 MVP (Matrix Vector Processor)",
    supported_ops=("matrix_vector_multiply",),
    supported_dtypes=("float16",),
    matrix_limits=None,
    alignment_bytes=None,
)

EFR32MG24_MVP = PlatformSpec(
    name="EFR32MG24 (MVP-accelerated)",
    flash_bytes=1536 * 1024,
    sram_bytes=256 * 1024,
    alignment_bytes=None,
    supported_numeric_types=("float32", "float16"),
    accelerator=EFR32MG24_MVP_ACCELERATOR,
    allocation_policy="dynamic",
)

EFR32MG24_CPU = PlatformSpec(
    name="EFR32MG24 (CPU-only)",
    flash_bytes=1536 * 1024,
    sram_bytes=256 * 1024,
    alignment_bytes=None,
    supported_numeric_types=("float32",),
    accelerator=None,
    allocation_policy="dynamic",
)
