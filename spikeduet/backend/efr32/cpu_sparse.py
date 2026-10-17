"""Event-driven Cortex-M33 affine kernel for the EFR32 (emitted C) and its cost model.

Provides the active-index scan helper and, per layer, a transposed FP32 weight table with a
register-blocked accumulate kernel. Binary inputs take an add-only path that is bitwise equal to
the dense scalar loop. This module does not import the IR, so the SNN backend and the MLP
baseline generator emit the same kernel.
"""
import math
from typing import Callable, List, Optional, Sequence

EFR32_CPU_SPARSE_BLOCK_ROWS = 16   # FPU register accumulators per block (s0-s15; s16-s31 hold weights)

# Cost per timestep at 39 MHz, binary inputs (least squares over in-firmware stage profiles,
# R^2 = 0.9998):
#   C = 12.15 * N/2  +  151.5 * ceil(M/16) * [step has an active input]  +  2.503 * active_inputs * M
# Calibrated on 20-64 output rows; it under-predicts 192-row layers by about 3-10%. Weights read
# from SRAM. Weights in flash add, per step:
#   5.55 * ceil(M/16) * [non-empty step] + 2.40 * active_inputs * M
# measured on paired flash/SRAM builds (R^2 = 0.99996). The general kernel for non-binary inputs is
# not modelled.
EFR32_CPU_SPARSE_SCAN_CYCLES_PER_PAIR = 12.15
EFR32_CPU_SPARSE_BLOCK_CYCLES = 151.5
EFR32_CPU_SPARSE_EVENT_ROW_CYCLES = 2.503
EFR32_CPU_SPARSE_FLASH_BLOCK_CYCLES = 5.55
EFR32_CPU_SPARSE_FLASH_WORD_CYCLES = 2.40


def cpu_sparse_affine_cycles(rows: int, cols: int, active_inputs: float, nonempty_fraction: float, *,
                             weights_in_sram: bool = True) -> float:
    """Mean cycles per step of one event-driven layer with ``active_inputs`` active inputs per step,
    ``nonempty_fraction`` of steps having at least one."""
    blocks = math.ceil(rows / EFR32_CPU_SPARSE_BLOCK_ROWS) * nonempty_fraction
    c = (EFR32_CPU_SPARSE_SCAN_CYCLES_PER_PAIR * cols / 2
         + EFR32_CPU_SPARSE_BLOCK_CYCLES * blocks
         + EFR32_CPU_SPARSE_EVENT_ROW_CYCLES * active_inputs * rows)
    if not weights_in_sram:
        c += EFR32_CPU_SPARSE_FLASH_BLOCK_CYCLES * blocks + EFR32_CPU_SPARSE_FLASH_WORD_CYCLES * active_inputs * rows
    return c

CPU_SPARSE_HELPERS = """
/* ---- event-driven sparse affine realization (Cortex-M33 FP32) ---- */
#if defined(__GNUC__) && !defined(__clang__)
#define ISCAS_CSP_OPT __attribute__((optimize("O2")))   /* the kernels only; the rest of the build keeps its -Os */
#else
#define ISCAS_CSP_OPT
#endif
static uint16_t *g_csp_idx[ISCAS_NUM_LAYERS];      /* active input indices of the current timestep */
static const float *g_csp_wT[ISCAS_NUM_LAYERS];    /* column-major FP32 W (flash table, or SRAM copy) */
static int g_csp_event_count[ISCAS_NUM_LAYERS];    /* active inputs processed */
static int g_csp_empty_count[ISCAS_NUM_LAYERS];    /* timesteps with no active input */
static int g_csp_nonbinary_count[ISCAS_NUM_LAYERS];/* timesteps that needed the general (x != 1) kernel */
static int g_csp_verify_mismatch[ISCAS_NUM_LAYERS];

/* Active-index discovery, one column pair per iteration (the same granularity as the MVP range builder).
 * *all_ones = every active value is exactly 1.0f. -0.0f counts as active (exact either way). */
ISCAS_CSP_OPT static int iscas_csp_scan(const float *src, int n, uint16_t *ai, int *all_ones)
{
    const uint32_t *xb = (const uint32_t *)src;
    uint32_t notone = 0u;
    int k = 0, i = 0;
    for (; i + 1 < n; i += 2) {
        const uint32_t a = xb[i], b = xb[i + 1];
        if ((a | b) != 0u) {
            if (a != 0u) { ai[k++] = (uint16_t)i; notone |= a ^ 0x3F800000u; }
            if (b != 0u) { ai[k++] = (uint16_t)(i + 1); notone |= b ^ 0x3F800000u; }
        }
    }
    if (i < n && xb[i] != 0u) { ai[k++] = (uint16_t)i; notone |= xb[i] ^ 0x3F800000u; }
    *all_ones = (notone == 0u);
    return k;
}
"""


def cpu_sparse_layer_definitions(idx: int, rows: int, cols: int, weight_rows: Sequence[Sequence[float]],
                                 bias_symbol: Optional[str], literal: Callable[[float], str], with_kernel: bool = True) -> List[str]:
    """Transposed weight table ``iscas_csp_wT{idx}`` and accumulate kernel ``iscas_csp_acc{idx}``.

    ``weight_rows`` is ``[rows][cols]``; ``bias_symbol`` names the C bias array or is ``None``;
    ``literal`` is the float formatter of the caller's ``model_data.h``, so both hold the same floats.
    """
    m, n = rows, cols
    out = [f"static const float iscas_csp_wT{idx}[{n * m}] = {{"]
    for c in range(n):
        out.append("    " + ", ".join(literal(weight_rows[r][c]) for r in range(m)) + ",")
    out.append("};")
    return out + (_acc_kernel_lines(idx, m, bias_symbol, pinned=False) if with_kernel else [])


def _acc_kernel_lines(idx: int, m: int, bias_symbol: Optional[str], pinned: bool) -> List[str]:
    """Register-blocked accumulate kernel. Column ``i`` starts at ``wT + i*M``, or at ``colp[i]``
    for a pinned layer; the arithmetic is identical."""
    name, arg = (f"iscas_csp_accp{idx}", "const float *const *colp") if pinned else (f"iscas_csp_acc{idx}", "const float *wT")
    out = [f"ISCAS_CSP_OPT static void {name}({arg}, const uint16_t *ai, int k, const float *xv, float *y)", "{"]
    for b0 in range(0, m, EFR32_CPU_SPARSE_BLOCK_ROWS):
        bw = min(EFR32_CPU_SPARSE_BLOCK_ROWS, m - b0)
        init = ", ".join(f"a{r} = " + (f"{bias_symbol}[{b0 + r}]" if bias_symbol else "0.0f") for r in range(bw))
        add1 = " ".join(f"a{r} += w[{r}];" for r in range(bw))
        addx = " ".join(f"a{r} += w[{r}] * x;" for r in range(bw))
        store = " ".join(f"y[{b0 + r}] = a{r};" for r in range(bw))
        base = "colp[ai[t]]" if pinned else f"wT + (uint32_t)ai[t] * {m}u"
        out += [
            f"    {{   /* rows {b0}..{b0 + bw - 1} */",
            f"        float {init};",
            "        if (xv == 0) {   /* binary events/spikes: acc + w == fma(w, 1, acc) exactly */",
            f"            for (int t = 0; t < k; t++) {{ const float *w = {base} + {b0}u; {add1} }}",
            "        } else {",
            f"            for (int t = 0; t < k; t++) {{ const float x = xv[ai[t]]; const float *w = {base} + {b0}u; {addx} }}",
            "        }",
            f"        {store}",
            "    }",
        ]
    out.append("}")
    return out


CPU_SPARSE_WEIGHT_PLACEMENTS = ("sram", "flash")


def cpu_sparse_init_lines(idx: int, cols: int, weights: Optional[str] = None) -> List[str]:
    """``snn_init()`` lines for one event-driven layer: the index list and the weight residency.
    ``weights=None`` leaves residency to the ``ISCAS_CPU_SPARSE_WT_SRAM`` define; ``"sram"`` or
    ``"flash"`` fixes it for this layer."""
    idx_line = f'    g_csp_idx[{idx}] = (uint16_t *)snn_alloc_check(malloc((size_t){cols} * sizeof(uint16_t)), "g_csp_idx");'
    if weights is not None:
        if weights not in CPU_SPARSE_WEIGHT_PLACEMENTS:
            raise ValueError(f"weights must be one of {CPU_SPARSE_WEIGHT_PLACEMENTS} or None, got {weights!r}")
        if weights == "flash":
            return [idx_line, f"    g_csp_wT[{idx}] = iscas_csp_wT{idx};   /* per-layer placement: read in place from flash */"]
        return [
            idx_line,
            "    {   /* per-layer placement: transposed weights copied to SRAM */",
            f'        float *wt = (float *)snn_alloc_check(malloc(sizeof(iscas_csp_wT{idx})), "g_csp_wT");',
            f"        memcpy(wt, iscas_csp_wT{idx}, sizeof(iscas_csp_wT{idx}));",
            f"        g_csp_wT[{idx}] = wt;",
            "    }",
        ]
    return [
        idx_line,
        "#ifdef ISCAS_CPU_SPARSE_WT_SRAM",
        "    {   /* optional: transposed weights copied to SRAM (default: read in place from flash) */",
        f'        float *wt = (float *)snn_alloc_check(malloc(sizeof(iscas_csp_wT{idx})), "g_csp_wT");',
        f"        memcpy(wt, iscas_csp_wT{idx}, sizeof(iscas_csp_wT{idx}));",
        f"        g_csp_wT[{idx}] = wt;",
        "    }",
        "#else",
        f"    g_csp_wT[{idx}] = iscas_csp_wT{idx};",
        "#endif",
    ]


# Hot-column pinning (opt-in): the layer's table stays in flash and a fixed set of columns, chosen
# from calibration activity with select_hot_columns, is copied to SRAM once in snn_init(). A
# per-column pointer table points each input column at its SRAM copy or its flash column. The
# additions are the same and in the same order, so results equal the unpinned kernel; nothing is
# copied during inference.
def select_hot_columns(column_counts: Sequence[float], n: int) -> List[int]:
    """The ``n`` most active columns (lower index wins ties), in ascending order. Use activity counts
    from a calibration split, not the evaluation split."""
    order = sorted(range(len(column_counts)), key=lambda c: (-column_counts[c], c))
    return sorted(order[:max(0, min(n, len(column_counts)))])


def cpu_sparse_pinned_definitions(idx: int, rows: int, cols: int, pinned: Sequence[int],
                                  bias_symbol: Optional[str]) -> List[str]:
    """Pinned-column list, pointer table and pointer-table kernel ``iscas_csp_accp{idx}``. The
    flash table itself comes from ``cpu_sparse_layer_definitions``."""
    pins = sorted(set(int(c) for c in pinned))
    if not pins or pins[0] < 0 or pins[-1] >= cols:
        raise ValueError(f"layer {idx}: pinned columns must be a non-empty subset of 0..{cols - 1}")
    return [f"static const uint16_t iscas_csp_pin{idx}[{len(pins)}] = {{" + ", ".join(map(str, pins)) + "};   /* hot columns, SRAM */",
            f"static const float *g_csp_colp{idx}[{cols}];   /* per-column weight pointer: SRAM copy or flash */",
            *_acc_kernel_lines(idx, rows, bias_symbol, pinned=True)]


def cpu_sparse_pinned_init_lines(idx: int, rows: int, cols: int, n_pinned: int) -> List[str]:
    return [
        "    {   /* hot-column pinning: copy the pinned columns to SRAM once; every other column is read from flash */",
        f'        float *pin = (float *)snn_alloc_check(malloc((size_t){n_pinned * rows} * sizeof(float)), "g_csp_pin");',
        f"        for (int c = 0; c < {cols}; c++) {{ g_csp_colp{idx}[c] = iscas_csp_wT{idx} + (uint32_t)c * {rows}u; }}",
        f"        for (int p = 0; p < {n_pinned}; p++) {{",
        f"            memcpy(pin + (uint32_t)p * {rows}u, g_csp_colp{idx}[iscas_csp_pin{idx}[p]], {rows}u * sizeof(float));",
        f"            g_csp_colp{idx}[iscas_csp_pin{idx}[p]] = pin + (uint32_t)p * {rows}u;",
        "        }",
        "    }",
    ]
