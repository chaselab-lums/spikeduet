"""Per-timestep cycle models of one EFR32 affine layer on the MVP, dense and range-sparse.

Calibrated on an EFR32MG24 at 39 MHz (DWT cycles, Simplicity SDK 2025.6.3). Anchors:

* SDK dense matvec: 22,851 cycles at 64x700 and 1,007 at 20x64, i.e.
  ``329 + M * (N/2 + 1.91)``;
* one range launch costs ``454 - 4.06 * (64 - M)`` (per-row accumulator spill and reload);
* range-sparse 64x700 input layer: ``5462 + 454 * launches + 87.9 * processed_pairs``;
* dense 64x700 input layer: copy 6.0 and cast 8.0 cycles per column, output loop 33.06 per row.

Shapes far from the anchors are extrapolations; the hidden-layer variants below were measured
separately on 64x64, 192x192 and 384x384 layers. The output loop is the same in the dense and
sparse realizations, so it cancels in every dense-versus-sparse decision.
"""

from spikeduet.backend.efr32.mvp_sparse import (
    EFR32_MVP_LAUNCH_CYCLES,
    EFR32_MVP_STAGE_CYCLES_PER_PAIR,
    fp32_merge_gap_pairs,
)


def sparse_merge_gap_pairs(rows: int, *, fp16_source: bool = False) -> int:
    """Largest inactive gap (in column pairs) worth processing instead of starting a new launch.

    Processing ``g`` extra pairs costs ``g * (rows + staging)`` cycles; a launch costs
    ``mvp_launch_cycles(rows)``. 64 rows with FP32 staging gives 5.
    """
    if fp16_source:
        return int(mvp_launch_cycles(rows)) // rows
    return fp32_merge_gap_pairs(rows)


EFR32_MVP_SDK_FIXED = 329.0
EFR32_MVP_SDK_ROW_OVERHEAD = 1.91
EFR32_MVP_LAUNCH_ROW_CYCLES = 4.06
EFR32_CPU_COPY_CYCLES_PER_COL = 6.0
EFR32_CPU_CAST_CYCLES_PER_COL = 8.0
EFR32_MVP_OUT_CYCLES_PER_ROW = 33.06
EFR32_MVP_SPARSE_SCAN_CYCLES_PER_PAIR = 8.57   # zero test of one column pair (range builder)
EFR32_MVP_SPARSE_ENTER_CYCLES = 346.0          # enter/setup/exit; with the scan: 5462 - 64*33.06 at 64x700


def mvp_launch_cycles(rows: int) -> float:
    """Cost of one extra range launch, incl. per-row accumulator spill/reload."""
    return EFR32_MVP_LAUNCH_CYCLES - EFR32_MVP_LAUNCH_ROW_CYCLES * (64 - rows)


def mvp_dense_affine_cycles(rows: int, cols: int, *, fp32_source: bool, copy: bool = False) -> float:
    """Dense MVP AffineOp per timestep: [copy] + [FP32->FP16 cast] + SDK matvec + output loop."""
    c = EFR32_MVP_SDK_FIXED + rows * (cols / 2 + EFR32_MVP_SDK_ROW_OVERHEAD) + EFR32_MVP_OUT_CYCLES_PER_ROW * rows
    if fp32_source:
        c += EFR32_CPU_CAST_CYCLES_PER_COL * cols + (EFR32_CPU_COPY_CYCLES_PER_COL * cols if copy else 0.0)
    return c


def mvp_sparse_affine_cycles(rows: int, cols: int, launches: float, pairs: float, *, fp32_source: bool,
                             flash_weights: bool = False) -> float:
    """Range-sparse MVP layer per step: ``launches`` merged ranges covering ``pairs`` column pairs.
    ``flash_weights``: FP16 weights read in place from flash (per-pair MVP work x1.47, launch x0.8)."""
    mvp_pair = rows * (1.47 if flash_weights else 1.0)
    per_pair = mvp_pair + (EFR32_MVP_STAGE_CYCLES_PER_PAIR if fp32_source else 0)
    launch = mvp_launch_cycles(rows) * (0.8 if flash_weights else 1.0)
    return (EFR32_MVP_SPARSE_SCAN_CYCLES_PER_PAIR * (cols // 2) + EFR32_MVP_SPARSE_ENTER_CYCLES
            + EFR32_MVP_OUT_CYCLES_PER_ROW * rows + launch * launches + per_pair * pairs)


# Hidden layers with static FP16 I/O (``mvp_static_io_layers``) have a cheaper output loop than the
# input layer: 14.0 cycles per row, a least-squares fit over five measured layers (64->20, 64->64,
# 64->35, 192->192, 192->27), each within 2.7%. Do not use mvp_dense_affine_cycles for hidden layers.
EFR32_MVP_HIDDEN_OUT_CYCLES_PER_ROW = 14.0


def mvp_dense_hidden_affine_cycles(rows: int, cols: int, *, staged: bool = False) -> float:
    """Dense MVP hidden layer with static I/O. ``staged``: the input is an FP32 buffer (an
    inhibited upstream LIF), which adds the CPU cast."""
    c = EFR32_MVP_SDK_FIXED + rows * (cols / 2 + EFR32_MVP_SDK_ROW_OVERHEAD) + EFR32_MVP_HIDDEN_OUT_CYCLES_PER_ROW * rows
    return c + (EFR32_CPU_CAST_CYCLES_PER_COL * cols if staged else 0.0)


# Range-sparse hidden spike edge (FP16 input in place, no staging), measured on 64x64 and 192x192
# layers with SRAM weights and 384x384 with flash weights: range scan 12.0 cycles per column pair
# plus 4.7 per active input, output loop 14.0 per row, mvp_launch_cycles(M) per launch, M cycles per
# processed pair. FP16 weights read from flash: per-pair cost x1.47, launch x0.8. A complete
# 384x384 build predicted 30.9k cycles per step and measured 30.3k.
EFR32_MVP_HIDDEN_SCAN_CYCLES_PER_PAIR = 12.0
EFR32_MVP_HIDDEN_SCAN_CYCLES_PER_EVENT = 4.7
EFR32_MVP_FLASH_PAIR_FACTOR = 1.47
EFR32_MVP_FLASH_LAUNCH_FACTOR = 0.8
# Putting a hidden edge on the MVP makes the upstream LIF write FP16 instead of FP32 spikes: +2.05
# cycles per upstream neuron per step. The placement planner adds it to MVP hidden candidates.
EFR32_MVP_FP16_SPIKE_WRITE_CYCLES = 2.05


def mvp_sparse_hidden_affine_cycles(rows: int, cols: int, launches: float, pairs: float, active_inputs: float, *,
                                    flash_weights: bool = False) -> float:
    """Range-sparse MVP hidden-edge AffineOp per timestep (see the constants above)."""
    launch = mvp_launch_cycles(rows) * (EFR32_MVP_FLASH_LAUNCH_FACTOR if flash_weights else 1.0)
    pair = rows * (EFR32_MVP_FLASH_PAIR_FACTOR if flash_weights else 1.0)
    return (EFR32_MVP_HIDDEN_SCAN_CYCLES_PER_PAIR * (cols // 2) + EFR32_MVP_HIDDEN_SCAN_CYCLES_PER_EVENT * active_inputs
            + EFR32_MVP_HIDDEN_OUT_CYCLES_PER_ROW * rows + launch * launches + pair * pairs)


EFR32_SPARSE_MIN_LAYER_GAIN = 0.10   # a profile-selected range-sparse layer must save at least 10%
