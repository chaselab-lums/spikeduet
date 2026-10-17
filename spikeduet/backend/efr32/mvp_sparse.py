"""Range-sparse MVP kernel for the EFR32 (emitted C) and its hardware constants.

The MVP cannot gather arbitrary columns (no indirect addressing, no predication, strictly affine
strides), so input sparsity is exploited by running the SDK matvec's own instruction sequence
(CLEAR; MACR2A over FP16 column pairs; ADDR) only over contiguous runs of active column pairs.
Runs separated by a small inactive gap are merged, and each row's (even, odd) lane accumulator is
carried between launches as an FP16 pair, which is lossless. Skipping an all-zero pair is exact,
so the result is bit-identical to ``sl_math_mvp_matrix_vector_mult_f16()``; ``ISCAS_SPARSE_VERIFY``
builds check this every step.

Measured on an EFR32MG24 (least squares, R^2 = 0.997) for 64 rows:
``cycles ~= 5462 + 454 * launches + 87.9 * processed_pairs`` (87.9 = 64 MVP cycles + ~24 CPU
staging cycles per pair).

This module does not import the IR, so the SNN backend and the MLP baseline generator emit the
same kernel.
"""

EFR32_MVP_LAUNCH_CYCLES = 454          # per extra MVP launch (setup + start + completion + per-row reload/spill)
EFR32_MVP_STAGE_CYCLES_PER_PAIR = 24   # CPU FP32->FP16 staging of one column pair
EFR32_MVP_SPARSE_MIN_LAUNCHES = 16     # dense MVP work must exceed this many launches for a scan to pay off
EFR32_MVP_MAX_DIM = 1024               # MVP array/loop dimension limit (10-bit SIZE/NUMITERS fields)


def fp32_merge_gap_pairs(rows: int) -> int:
    """Merge gap for an FP32-staged source: ``launch // (rows + staging)``; 5 for 64 rows."""
    return EFR32_MVP_LAUNCH_CYCLES // (rows + EFR32_MVP_STAGE_CYCLES_PER_PAIR)


# Raw-register MVP helpers, emitted only when a layer is range-sparse. The register programming
# mirrors sl_math_mvp_matrix_vector_mult_f16() (Simplicity SDK 2025.6.3), restricted to a column slice.
SPARSE_MVP_HELPERS = """
/* ---- range-sparse MVP realization ---------------- */
#define ISCAS_SP_T16   ((uint32_t)SLI_MVP_DATATYPE_BINARY16 << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT)
#define ISCAS_SP_T16C  ((uint32_t)SLI_MVP_DATATYPE_COMPLEX_BINARY16 << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT)
#define ISCAS_SP_DIM(size, stride) ((((uint32_t)(size) - 1u) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | ((uint32_t)(stride) << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT))
#define ISCAS_SP_FAULTS (MVP_IF_LOOPFAULT | MVP_IF_BUSERRFAULT | MVP_IF_BUSALIGNFAULT | MVP_IF_ALUFAULT | MVP_IF_ARRAYFAULT)
#define ISCAS_SP_LAUNCH_CYCLES %(launch)d
#define ISCAS_SP_STAGE_CYCLES_PER_PAIR %(stage)d
enum { ISCAS_SP_A_W = 0, ISCAS_SP_A_S = 1, ISCAS_SP_A_OUT = 2, ISCAS_SP_A_ACC = 3 };

static uint32_t g_sparse_saved_ien;
static int g_sparse_mvp_fault;

static void iscas_sp_enter(void)
{
    sli_mvp_cmd_wait_for_completion();
    g_sparse_saved_ien = MVP->IEN;
    MVP->IEN_CLR = 0xFFFFFFFFu;   /* completion is polled below: no ISR, no power-manager bookkeeping */
    MVP->IF_CLR = 0xFFFFFFFFu;
}

static void iscas_sp_exit(void)
{
    MVP->IF_CLR = 0xFFFFFFFFu;
    MVP->IEN_SET = g_sparse_saved_ien;
}

static void iscas_sp_setup(int rows, int rowstride_pairs, float16_t *out, float16_t *acc)
{
    MVP->ARRAY[ISCAS_SP_A_W].DIM0CFG = ISCAS_SP_T16C;
    MVP->ARRAY[ISCAS_SP_A_W].DIM1CFG = ISCAS_SP_DIM(rows, rowstride_pairs);
    MVP->ARRAY[ISCAS_SP_A_S].DIM0CFG = ISCAS_SP_T16C;
    MVP->ARRAY[ISCAS_SP_A_OUT].DIM0CFG = ISCAS_SP_T16;
    MVP->ARRAY[ISCAS_SP_A_OUT].DIM1CFG = ISCAS_SP_DIM(1, rows);
    MVP->ARRAY[ISCAS_SP_A_OUT].DIM2CFG = ISCAS_SP_DIM(rows, 1);
    MVP->ARRAY[ISCAS_SP_A_OUT].ADDRCFG = (uint32_t)out;
    MVP->ARRAY[ISCAS_SP_A_ACC].DIM0CFG = ISCAS_SP_T16C;
    MVP->ARRAY[ISCAS_SP_A_ACC].DIM1CFG = ISCAS_SP_DIM(1, rows);
    MVP->ARRAY[ISCAS_SP_A_ACC].DIM2CFG = ISCAS_SP_DIM(rows, 1);
    MVP->ARRAY[ISCAS_SP_A_ACC].ADDRCFG = (uint32_t)acc;
    MVP->LOOP[1].RST = 0;
    MVP->LOOP[0].CFG = ((uint32_t)(rows - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT);
    MVP->INSTR[1].CFG0 = SLI_MVP_ALU_X(SLI_MVP_R1) | SLI_MVP_ALU_Y(SLI_MVP_R2) | SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3);
    MVP->INSTR[1].CFG1 = SLI_MVP_LOAD(0, SLI_MVP_R1, ISCAS_SP_A_W, SLI_MVP_INCRDIM_COL) | SLI_MVP_LOAD(1, SLI_MVP_R2, ISCAS_SP_A_S, SLI_MVP_INCRDIM_COL);
    MVP->INSTR[1].CFG2 = (SLI_MVP_OP(MACR2A) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP1BEGIN | MVP_INSTRCFG2_LOOP1END;
}

/* One launch over column pairs [p0, p0+wp). first: lanes start from CLEAR (as the SDK kernel),
 * otherwise reloaded from ACC. last: lanes merged with ADDR and stored as real FP16 to OUT (as the
 * SDK kernel), otherwise spilled back to ACC. In the last launch the reload itself must advance the
 * ACC index, because the store goes to OUT (omitting this made every row reload row 0). */
static void iscas_sp_launch(const float16_t *W, const float16_t *s, int p0, int wp, int first, int last)
{
    MVP->ARRAY[ISCAS_SP_A_W].ADDRCFG = (uint32_t)(W + 2 * p0);
    MVP->ARRAY[ISCAS_SP_A_W].DIM2CFG = ISCAS_SP_DIM(wp, 1);
    MVP->ARRAY[ISCAS_SP_A_S].ADDRCFG = (uint32_t)(s + 2 * p0);
    MVP->ARRAY[ISCAS_SP_A_S].DIM1CFG = ISCAS_SP_DIM(1, wp);
    MVP->ARRAY[ISCAS_SP_A_S].DIM2CFG = ISCAS_SP_DIM(wp, 1);
    MVP->LOOP[1].CFG = ((uint32_t)(wp - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                       | ((uint32_t)SLI_MVP_LOOP_INCRDIM(ISCAS_SP_A_W, SLI_MVP_INCRDIM_ROW) << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);
    if (first) {
        MVP->INSTR[0].CFG0 = SLI_MVP_ALU_Z(SLI_MVP_R3);
        MVP->INSTR[0].CFG1 = 0;
        MVP->INSTR[0].CFG2 = (SLI_MVP_OP(CLEAR) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP0BEGIN;
    } else {
        MVP->INSTR[0].CFG0 = 0;
        MVP->INSTR[0].CFG1 = SLI_MVP_LOAD(0, SLI_MVP_R3, ISCAS_SP_A_ACC, last ? SLI_MVP_INCRDIM_COL : 0);
        MVP->INSTR[0].CFG2 = (SLI_MVP_OP(NOOP) << _MVP_INSTRCFG2_ALUOP_SHIFT) | MVP_INSTRCFG2_LOOP0BEGIN;
    }
    MVP->INSTR[2].CFG0 = last ? (SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3)) : 0;
    MVP->INSTR[2].CFG1 = last ? SLI_MVP_STORE(SLI_MVP_R3, ISCAS_SP_A_OUT, SLI_MVP_INCRDIM_COL)
                              : SLI_MVP_STORE(SLI_MVP_R3, ISCAS_SP_A_ACC, SLI_MVP_INCRDIM_COL);
    MVP->INSTR[2].CFG2 = ((last ? SLI_MVP_OP(ADDR) : SLI_MVP_OP(NOOP)) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                         | MVP_INSTRCFG2_LOOP0END | MVP_INSTRCFG2_ENDPROG;
    MVP->CMD = MVP_CMD_INIT | MVP_CMD_START;
    uint32_t f;
    while (!((f = MVP->IF) & (MVP_IF_PROGDONE | ISCAS_SP_FAULTS))) {
    }
    if (f & ISCAS_SP_FAULTS) { g_sparse_mvp_fault = 1; }
    MVP->IF_CLR = 0xFFFFFFFFu;
}
""" % {"launch": EFR32_MVP_LAUNCH_CYCLES, "stage": EFR32_MVP_STAGE_CYCLES_PER_PAIR}


# Asynchronous range launches (experimental, ``async_mvp=True``), emitted after SPARSE_MVP_HELPERS.
# A launch is split into a start (the same register programming, no wait), a completion test and an
# acknowledge. A chain of range launches is then advanced by iscas_sp_poll() calls placed inside the
# Cortex-M33's own loops; the output is bit-identical to the blocking chain. The hardware part and the
# hardware-free chain logic are separate text blocks so host tests can replace the former:
#   ASYNC_MVP_HW_HELPERS     iscas_sp_start / iscas_sp_flags / iscas_sp_ack
#   ASYNC_MVP_CHAIN_HELPERS  iscas_sp_begin_chain / iscas_sp_poll / iscas_sp_wait_chain
def _start_from_launch() -> str:
    """iscas_sp_start() == iscas_sp_launch() up to and including the CMD write (one source for the register
    programming), with a memory barrier before the start."""
    _head, rest = SPARSE_MVP_HELPERS.split("static void iscas_sp_launch(")
    body, _tail = rest.split("    uint32_t f;")
    cmd = "    MVP->CMD = MVP_CMD_INIT | MVP_CMD_START;"
    assert body.rstrip().endswith(cmd), "iscas_sp_launch changed: update _start_from_launch"
    body = body.rstrip()[: -len(cmd)]
    return ("static void iscas_sp_start(" + body
            + "    __DMB();   /* every operand the program reads (spike vector, accumulators) is written before it starts */\n"
            + cmd + "\n}\n")


ASYNC_MVP_HW_HELPERS = """
/* ---- asynchronous range launches: hardware part ---- */
#ifdef ISCAS_ASYNC_PERF
static uint32_t g_async_run, g_async_stall, g_async_launches;   /* the MVP's own RUN / STALL counters, summed per inference */
#endif
/* One launch over column pairs [p0, p0+wp): the register programming of iscas_sp_launch(), without the wait. */
""" + _start_from_launch() + """
static inline uint32_t iscas_sp_flags(void) { return MVP->IF & (MVP_IF_PROGDONE | ISCAS_SP_FAULTS); }

static inline void iscas_sp_ack(uint32_t f)
{
    if (f & ISCAS_SP_FAULTS) { g_sparse_mvp_fault = 1; }
#ifdef ISCAS_ASYNC_PERF
    g_async_run += MVP->PERF[0].CNT; g_async_stall += MVP->PERF[1].CNT; g_async_launches++;
#endif
    MVP->IF_CLR = 0xFFFFFFFFu;
}
"""

_ASYNC_CHAIN_BASE = """
/* ---- asynchronous range launches: chain of merged ranges, advanced by polling (hardware-free) ---- */
static struct {
    const float16_t *W, *s;
    const uint16_t *rs, *re;
    int nr, next;
    int active;          /* a launch is in flight, or more ranges remain */
} g_async;

/* Starts range 0; the caller has run iscas_sp_enter() and iscas_sp_setup(). nr >= 1. */
static void iscas_sp_begin_chain(const float16_t *W, const float16_t *s, const uint16_t *rs, const uint16_t *re, int nr)
{
    g_async.W = W; g_async.s = s; g_async.rs = rs; g_async.re = re; g_async.nr = nr; g_async.next = 1;
    iscas_sp_start(W, s, rs[0], re[0] - rs[0], 1, nr == 1);
    g_async.active = 1;
}

/* Called from the Cortex-M33's loops while the chain runs: if the launch in flight has completed (or faulted),
 * acknowledge it and start the next range. A fault is recorded and the chain still runs to its end, so the caller
 * never waits on a program that will not complete. */
static inline void iscas_sp_poll(void)
{
    if (!g_async.active) { return; }
    const uint32_t f = iscas_sp_flags();
    if (!f) { return; }
    iscas_sp_ack(f);
    if (g_async.next < g_async.nr) {
        const int r = g_async.next++;
        iscas_sp_start(g_async.W, g_async.s, g_async.rs[r], g_async.re[r] - g_async.rs[r], 0, r == g_async.nr - 1);
    } else {
        g_async.active = 0;
    }
}

"""

_ASYNC_WAIT_POLLED = """/* Blocks until every range has run. After it returns the MVP is idle and its output vector is complete. */
static void iscas_sp_wait_chain(void)
{
    while (g_async.active) { iscas_sp_poll(); }
    __DMB();
}
"""
_ASYNC_WAIT_REGISTER_ONLY = """/* Blocks until every range has run. After it returns the MVP is idle and its output vector is complete.
 * Register-only variant (async_fast_wait): the remaining ranges are issued from locals with the blocking launch's
 * own tight completion loop, so the Cortex-M33 reads no memory while it waits for a program. */
static void iscas_sp_wait_chain(void)
{
    if (g_async.active) {
        const float16_t *const W = g_async.W, *const s = g_async.s;
        const uint16_t *const rs = g_async.rs, *const re = g_async.re;
        const int nr = g_async.nr;
        int next = g_async.next;
        for (;;) {
            uint32_t f;
            while (!(f = iscas_sp_flags())) {
            }
            iscas_sp_ack(f);
            if (next >= nr) { break; }
            iscas_sp_start(W, s, rs[next], (int)re[next] - (int)rs[next], 0, next == nr - 1);
            next++;
        }
        g_async.next = next;
        g_async.active = 0;
    }
    __DMB();
}
"""
ASYNC_MVP_CHAIN_HELPERS = _ASYNC_CHAIN_BASE + _ASYNC_WAIT_POLLED
ASYNC_MVP_CHAIN_HELPERS_FAST_WAIT = _ASYNC_CHAIN_BASE + _ASYNC_WAIT_REGISTER_ONLY   # async_fast_wait

# Host stand-in for the hardware helpers (tests and host reference runs only). The range kernel is
# modelled as two FP16 lane accumulators per row (even / odd column) added at the end, which matches
# the device bit for bit. A started launch completes on the HOST_ASYNC_DONE_AFTER-th completion test
# and reads its operands only then, so a buffer hazard in the generated schedule changes the result.
HOST_RANGE_KERNEL_STUB = """
/* ---- HOST twin of the range kernel (tests / host reference only) ---- */
#ifndef HOST_ASYNC_DONE_AFTER
#define HOST_ASYNC_DONE_AFTER 3
#endif
#define ISCAS_SP_LAUNCH_CYCLES %(launch)d
#define ISCAS_SP_STAGE_CYCLES_PER_PAIR %(stage)d
static int g_sparse_mvp_fault;
static struct { int rows, rsp; float16_t *out, *acc; const float16_t *W, *s; int p0, wp, first, last, pending, tests; } g_host;
static void iscas_sp_enter(void) {}
static void iscas_sp_exit(void) {}
static void iscas_sp_setup(int rows, int rowstride_pairs, float16_t *out, float16_t *acc)
{ g_host.rows = rows; g_host.rsp = rowstride_pairs; g_host.out = out; g_host.acc = acc; }
static void host_range(const float16_t *W, const float16_t *s, int p0, int wp, int first, int last)
{
    for (int r = 0; r < g_host.rows; r++) {
        float16_t a0 = first ? (float16_t)0 : g_host.acc[2 * r], a1 = first ? (float16_t)0 : g_host.acc[2 * r + 1];
        const float16_t *w = W + (size_t)r * 2 * g_host.rsp;
        for (int p = p0; p < p0 + wp; p++) {
            a0 = (float16_t)((double)a0 + (double)(float16_t)((double)w[2 * p] * (double)s[2 * p]));
            a1 = (float16_t)((double)a1 + (double)(float16_t)((double)w[2 * p + 1] * (double)s[2 * p + 1]));
        }
        if (last) { g_host.out[r] = (float16_t)((double)a0 + (double)a1); }
        else { g_host.acc[2 * r] = a0; g_host.acc[2 * r + 1] = a1; }
    }
}
static void iscas_sp_launch(const float16_t *W, const float16_t *s, int p0, int wp, int first, int last) { host_range(W, s, p0, wp, first, last); }
static void iscas_sp_start(const float16_t *W, const float16_t *s, int p0, int wp, int first, int last)
{ g_host.W = W; g_host.s = s; g_host.p0 = p0; g_host.wp = wp; g_host.first = first; g_host.last = last; g_host.pending = 1; g_host.tests = 0; }
static inline uint32_t iscas_sp_flags(void)
{
    if (!g_host.pending || ++g_host.tests < HOST_ASYNC_DONE_AFTER) { return 0u; }
    host_range(g_host.W, g_host.s, g_host.p0, g_host.wp, g_host.first, g_host.last);
    g_host.pending = 0;
    return 1u;
}
static inline void iscas_sp_ack(uint32_t f) { (void)f; }
#define __DMB() ((void)0)
""" % {"launch": EFR32_MVP_LAUNCH_CYCLES, "stage": EFR32_MVP_STAGE_CYCLES_PER_PAIR}
