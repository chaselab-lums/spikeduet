/* app.c -- UART and timing shell for SpikeDuet EFR32 firmware.
 *
 * Model independent: the input buffer is sized from ISCAS_INPUT_DIM (model_data.h) and the model
 * is reached only through the engine contract in snn.h. ISCAS_INPUT_TEMPORAL selects one of two
 * wire protocols at compile time (see the README):
 *
 *   static   MCU_READY, then per inference ISCAS_INPUT_DIM integers scaled by 10000 and a
 *            PRED:<class>:<infer_us>:<parse_us>:<total_us> reply;
 *   temporal BEGIN_TEMPORAL <steps> <dim> / READY_T:0, then one vector per step, each
 *            acknowledged with ACK_T:<t>, then PRED.
 *
 * infer_us covers the model computation only (DWT cycles). Optional ISCAS_* report blocks print
 * after PRED, outside every timed region.
 */
#include "app.h"
/* model_data.h must come before snn.h (snn.h uses iscas_layer_t). */
#include "model_data.h"
#include "snn.h"
#include "em_device.h"
#include "sl_iostream.h"
#include "sl_status.h"
#include <stdio.h>
#include <stdint.h>
#ifdef ISCAS_ENERGY_REPLAY
#include <stdlib.h>
#endif

#define INPUT_SCALE 10000.0f

/* A lost UART byte must not hang the board: every byte read times out, and the board answers
 * ERR:UART_TIMEOUT instead. The bound is far above any legitimate inter-byte gap. */
#define ISCAS_UART_BYTE_TIMEOUT_MS 2000U
/* After a timeout, drain bytes until the line has been quiet this long, so a partial request does
 * not desync the next one. */
#define ISCAS_UART_DRAIN_QUIET_MS 100U

static uint32_t ms_to_cycles(uint32_t ms)
{
    return (uint32_t)(((uint64_t)ms * SystemCoreClock) / 1000ULL);
}

/* Reads one byte, polling sl_iostream_getchar() for at most ISCAS_UART_BYTE_TIMEOUT_MS.
 * Returns 0 on timeout. The DWT delta is wrap-safe because the timeout is far shorter than one
 * CYCCNT period. */
static int uart_getchar_timeout(char *out_c)
{
    uint32_t start = DWT->CYCCNT;
    uint32_t timeout_cycles = ms_to_cycles(ISCAS_UART_BYTE_TIMEOUT_MS);

    while ((uint32_t)(DWT->CYCCNT - start) < timeout_cycles) {
        sl_status_t status = sl_iostream_getchar(SL_IOSTREAM_STDIN, out_c);
        if (status == SL_STATUS_OK) {
            return 1;
        }
        /* SL_STATUS_EMPTY (nothing queued yet) -- keep polling. */
    }
    return 0;
}

/* Best-effort flush after a timeout (see ISCAS_UART_DRAIN_QUIET_MS). */
static void uart_drain_stray_bytes(void)
{
    char c;
    uint32_t quiet_start = DWT->CYCCNT;
    uint32_t quiet_cycles = ms_to_cycles(ISCAS_UART_DRAIN_QUIET_MS);

    while ((uint32_t)(DWT->CYCCNT - quiet_start) < quiet_cycles) {
        if (sl_iostream_getchar(SL_IOSTREAM_STDIN, &c) == SL_STATUS_OK) {
            quiet_start = DWT->CYCCNT;  /* saw a byte -- extend the quiet window */
        }
    }
}

/* Parses one signed integer token, skipping separators before it. Returns 0 if a byte timed out
 * mid-request; callers must check. */
static int read_robust_int(int *out_val)
{
    int val = 0;
    int sign = 1;
    char c;
    int started = 0;

    while (1) {
        if (!uart_getchar_timeout(&c)) {
            return 0;
        }

        if (!started) {
            if (c == '-') {
                sign = -1;
                started = 1;
            } else if (c >= '0' && c <= '9') {
                val = c - '0';
                started = 1;
            }
            /* any other byte before a token is a separator and is skipped */
        } else {
            if (c >= '0' && c <= '9') {
                val = (val * 10) + (c - '0');
            } else {
                break;
            }
        }
    }

    *out_val = val * sign;
    return 1;
}

#ifdef ISCAS_ENERGY_REPLAY
/* Energy replay (default off): see the block at the end of app_process_action(). */
#ifndef ISCAS_REPLAY_TARGET_MS
#define ISCAS_REPLAY_TARGET_MS 600U      /* burst length the replay count is chosen for */
#endif
#ifndef ISCAS_REPLAY_GAP_MS
#define ISCAS_REPLAY_GAP_MS 15U          /* EM1 marker gap before and after the burst */
#endif
#define ISCAS_REPLAY_CMD_TIMEOUT_MS 40000U
#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
#define ISCAS_REPLAY_BYTES ((ISCAS_INPUT_DIM + 7) / 8)
/* Allocated AFTER snn_init() (see app_init) so every model buffer keeps the address it has without replay: the
 * MVP's speed depends on operand placement, so a static array here would perturb the thing being measured. */
static uint8_t *g_replay_bits;
static int g_replay_nonbinary;
#define REPLAY_ROW(t) (g_replay_bits + (size_t)(t) * ISCAS_REPLAY_BYTES)
#endif

void SysTick_Handler(void)
{
}

static void replay_gap_ms(uint32_t ms)
{
    SysTick->LOAD = ms_to_cycles(ms) - 1U;
    SysTick->VAL = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
    while (!(SysTick->CTRL & SysTick_CTRL_COUNTFLAG_Msk)) {
        __WFI();
    }
    SysTick->CTRL = 0U;
}

static int replay_get_cmd(char *out_c)
{
    uint32_t start = DWT->CYCCNT;
    uint32_t timeout_cycles = ms_to_cycles(ISCAS_REPLAY_CMD_TIMEOUT_MS);
    while ((uint32_t)(DWT->CYCCNT - start) < timeout_cycles) {
        if (sl_iostream_getchar(SL_IOSTREAM_STDIN, out_c) == SL_STATUS_OK) {
            return 1;
        }
    }
    return 0;
}
#endif

static void dwt_init_once(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(DWT_LAR)
    DWT->LAR = 0xC5ACCE55UL;
#endif
    DWT->CYCCNT = 0UL;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t cycles_to_us(uint32_t cycles)
{
    uint32_t hz = SystemCoreClock;
    if (hz == 0U) {
        hz = 39000000U;
    }
    return (uint32_t)(((uint64_t)cycles * 1000000ULL + (hz / 2U)) / hz);
}

#ifdef ISCAS_REPORT_MEMINFO
/* Measured runtime SRAM (default off). malloc() is newlib-nano growing the heap with sbrk(), and
 * nothing is ever freed, so mallinfo().arena = peak heap taken from the heap region, chunk overhead included
 * (uordblks = bytes currently in use). The unused stack below the current SP is painted at boot; the deepest
 * overwritten word gives the stack high-water mark. Printed after PRED, outside every DWT bracket. */
#include <malloc.h>
#include "sl_memory_manager_region.h"
#define ISCAS_STACK_PAINT 0xA5C3A5C3u
static void iscas_paint_stack(void)
{
    const sl_memory_region_t st = sl_memory_get_stack_region();
    uint32_t *p = (uint32_t *)st.addr;
    uint32_t *stop = (uint32_t *)(__get_MSP() - 64u);
    while (p < stop) { *p++ = ISCAS_STACK_PAINT; }
}
static void iscas_print_meminfo(void)
{
    const sl_memory_region_t st = sl_memory_get_stack_region();
    const uint32_t *p = (const uint32_t *)st.addr;
    const uint32_t *end = (const uint32_t *)((const uint8_t *)st.addr + st.size);
    while (p < end && *p == ISCAS_STACK_PAINT) { p++; }
    const struct mallinfo mi = mallinfo();
    const sl_memory_region_t hp = sl_memory_get_heap_region();
    printf("MEMINFO:heap_total=%lu:heap_used=%lu:heap_hwm=%lu:stack_size=%lu:stack_hwm=%lu\n",
           (unsigned long)hp.size, (unsigned long)mi.uordblks, (unsigned long)mi.arena, (unsigned long)st.size,
           (unsigned long)((const uint8_t *)end - (const uint8_t *)p));
}
#endif

void app_init(void)
{
#ifdef ISCAS_REPORT_MEMINFO
    iscas_paint_stack();
#endif
    dwt_init_once();
    snn_init();
#if defined(ISCAS_ENERGY_REPLAY) && defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
    g_replay_bits = (uint8_t *)malloc((size_t)ISCAS_STEPS * ISCAS_REPLAY_BYTES);
    if (g_replay_bits == NULL) {
        printf("ALLOC_FAIL:replay_bits\n");
        for (;;) {
        }
    }
#endif

    printf("MCU_READY\n");
}

void app_process_action(void)
{
#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
    /* Temporal protocol: each step's vector is acknowledged before the host sends the next, which
     * bounds every round trip to one step and localizes a lost byte to ERR:UART_TIMEOUT:t=<t>:i=<i>.
     *
     *   host -> "BEGIN_TEMPORAL <steps> <dim>"   MCU -> "READY_T:0" (or ERR:BAD_HEADER)
     *   host -> <ISCAS_INPUT_DIM values>         MCU -> "ACK_T:<t>"      (for every step t)
     *                                            MCU -> "PRED:<class>:<infer_us>:<parse_us>:<total_us>"
     *
     * The word BEGIN_TEMPORAL is only a marker (the token parser skips non-digits); the two
     * numbers are checked against the compiled ISCAS_STEPS / ISCAS_INPUT_DIM. Each vector is
     * consumed by iscas_infer_step() before the buffer is reused, so only one step is ever
     * resident. ACK_T is printed outside both DWT brackets and never counts as parse or infer time. */
    static float step_buf[ISCAS_INPUT_DIM];
    uint32_t parse_cycles = 0;
    uint32_t infer_cycles = 0;

    {
        int steps_hdr, dim_hdr;
        if (!read_robust_int(&steps_hdr) || !read_robust_int(&dim_hdr)) {
            uart_drain_stray_bytes();
            printf("ERR:UART_TIMEOUT:header\n");
            return;
        }
        if (steps_hdr != ISCAS_STEPS || dim_hdr != ISCAS_INPUT_DIM) {
            uart_drain_stray_bytes();
            printf("ERR:BAD_HEADER:got_steps=%d:got_dim=%d:expected_steps=%d:expected_dim=%d\n",
                   steps_hdr, dim_hdr, ISCAS_STEPS, ISCAS_INPUT_DIM);
            return;
        }
    }
    printf("READY_T:0\n");
#ifdef ISCAS_ENERGY_REPLAY
    g_replay_nonbinary = 0;
#endif

    for (int t = 0; t < ISCAS_STEPS; t++) {
        uint32_t p0 = DWT->CYCCNT;
        for (int i = 0; i < ISCAS_INPUT_DIM; i++) {
            int scaled_val;
            if (!read_robust_int(&scaled_val)) {
                uart_drain_stray_bytes();
                printf("ERR:UART_TIMEOUT:t=%d:i=%d\n", t, i);
                return;
            }
            step_buf[i] = (float)scaled_val / INPUT_SCALE;
        }
        uint32_t p1 = DWT->CYCCNT;
        parse_cycles += (p1 - p0);

#ifdef ISCAS_ENERGY_REPLAY
        /* Energy replay: keep this frame bit-packed so the replay block can re-run the same
         * inference. Outside both DWT brackets. */
        uint8_t *replay_row = REPLAY_ROW(t);
        for (int b = 0; b < ISCAS_REPLAY_BYTES; b++) {
            replay_row[b] = 0u;
        }
        for (int i = 0; i < ISCAS_INPUT_DIM; i++) {
            if (step_buf[i] != 0.0f) {
                if (step_buf[i] != 1.0f) {
                    g_replay_nonbinary = 1;
                }
                replay_row[i >> 3] |= (uint8_t)(1u << (i & 7));
            }
        }
#endif

        uint32_t i0 = DWT->CYCCNT;
        if (t == 0) {
            iscas_infer_reset(step_buf);
        }
        iscas_infer_step(step_buf);
        uint32_t i1 = DWT->CYCCNT;
        infer_cycles += (i1 - i0);

        /* Outside both DWT brackets: ACK transmission is not parse or infer time. */
        printf("ACK_T:%d\n", t);
    }

    uint32_t r0 = DWT->CYCCNT;
    int prediction = iscas_infer_readout();
    uint32_t r1 = DWT->CYCCNT;
    infer_cycles += (r1 - r0);

    uint32_t parse_us = cycles_to_us(parse_cycles);
    uint32_t infer_us = cycles_to_us(infer_cycles);
    uint32_t total_us = parse_us + infer_us;
#else
    float input_buffer[ISCAS_INPUT_DIM];

    uint32_t parse_start = DWT->CYCCNT;

    for (int i = 0; i < ISCAS_INPUT_DIM; i++) {
        int scaled_val;
        if (!read_robust_int(&scaled_val)) {
            uart_drain_stray_bytes();
            printf("ERR:UART_TIMEOUT:i=%d\n", i);
            return;
        }
        input_buffer[i] = (float)scaled_val / INPUT_SCALE;
    }

    uint32_t parse_end = DWT->CYCCNT;
    uint32_t infer_start = DWT->CYCCNT;

    int prediction = run_snn_inference(input_buffer);

    uint32_t infer_end = DWT->CYCCNT;

    uint32_t parse_us = cycles_to_us(parse_end - parse_start);
    uint32_t infer_us = cycles_to_us(infer_end - infer_start);
    uint32_t total_us = cycles_to_us(infer_end - parse_start);
#endif

    printf("PRED:%d:%lu:%lu:%lu\n",
           prediction,
           (unsigned long)infer_us,
           (unsigned long)parse_us,
           (unsigned long)total_us);

#ifdef ISCAS_COUNT_EXEC_GEN
    /* Per-layer op execution counts (iscas_debug_*_exec_count() in the generated engine). */
    extern int iscas_debug_affine_exec_count(int layer);
    extern int iscas_debug_lif_exec_count(int layer);
    extern int iscas_debug_convert_exec_count(int layer);
#ifdef ISCAS_COUNT_EXEC_GEN_MVP
    extern int iscas_debug_mvp_stage_cast_count(int layer);
    extern int iscas_debug_mvp_direct_fp16_write_count(int layer);
#endif
    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {
#ifdef ISCAS_COUNT_EXEC_GEN_MVP
        printf("EXECCOUNT:%d:affine=%d:lif=%d:convert=%d:stagecast=%d:directfp16=%d\n", l,
               iscas_debug_affine_exec_count(l), iscas_debug_lif_exec_count(l),
               iscas_debug_convert_exec_count(l), iscas_debug_mvp_stage_cast_count(l),
               iscas_debug_mvp_direct_fp16_write_count(l));
#else
        printf("EXECCOUNT:%d:affine=%d:lif=%d:convert=%d\n", l,
               iscas_debug_affine_exec_count(l), iscas_debug_lif_exec_count(l),
               iscas_debug_convert_exec_count(l));
#endif
    }
#endif

#ifdef ISCAS_COUNT_SPARSE_GEN
    /* Range-sparse MVP counters (sparse_mvp=True builds). Per inference: MVP launches, processed column pairs,
     * merged ranges, empty timesteps, cost-model fallbacks, and (ISCAS_SPARSE_VERIFY
     * correctness builds only) steps whose output differed bitwise from the dense kernel.
     * RAWCYCLES reports the raw DWT inference cycles and the runtime core clock. */
    extern int iscas_debug_sparse_layer(int layer);
    extern int iscas_debug_sparse_launch_count(int layer);
    extern int iscas_debug_sparse_pair_count(int layer);
    extern int iscas_debug_sparse_range_count(int layer);
    extern int iscas_debug_sparse_empty_count(int layer);
    extern int iscas_debug_sparse_fallback_count(int layer);
    extern int iscas_debug_sparse_verify_mismatch(int layer);
    extern int iscas_debug_sparse_mvp_fault(void);
    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {
        if (!iscas_debug_sparse_layer(l)) {
            continue;
        }
        printf("SPARSECOUNT:%d:launches=%d:pairs=%d:ranges=%d:empty=%d:fallback=%d:verify_mismatch=%d:fault=%d\n", l,
               iscas_debug_sparse_launch_count(l), iscas_debug_sparse_pair_count(l),
               iscas_debug_sparse_range_count(l), iscas_debug_sparse_empty_count(l),
               iscas_debug_sparse_fallback_count(l), iscas_debug_sparse_verify_mismatch(l),
               iscas_debug_sparse_mvp_fault());
    }
#if ISCAS_INPUT_TEMPORAL
    printf("RAWCYCLES:infer=%lu:clock_hz=%lu\n", (unsigned long)infer_cycles, (unsigned long)SystemCoreClock);
#endif
#endif

#ifdef ISCAS_COUNT_CPU_SPARSE_GEN
    /* Event-driven Cortex-M33 counters (sparse_cpu / cpu_sparse_layers builds): active inputs
     * processed, empty timesteps, timesteps that needed the general
     * (non-binary) kernel, and (ISCAS_SPARSE_VERIFY builds only) timesteps whose output differed
     * bitwise from the dense scalar recompute. */
    extern int iscas_debug_cpu_sparse_layer(int layer);
    extern int iscas_debug_cpu_sparse_event_count(int layer);
    extern int iscas_debug_cpu_sparse_empty_count(int layer);
    extern int iscas_debug_cpu_sparse_nonbinary_count(int layer);
    extern int iscas_debug_cpu_sparse_verify_mismatch(int layer);
    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) {
        if (!iscas_debug_cpu_sparse_layer(l)) {
            continue;
        }
        printf("CPUSPARSE:%d:events=%d:empty=%d:nonbinary=%d:verify_mismatch=%d\n", l,
               iscas_debug_cpu_sparse_event_count(l), iscas_debug_cpu_sparse_empty_count(l),
               iscas_debug_cpu_sparse_nonbinary_count(l), iscas_debug_cpu_sparse_verify_mismatch(l));
    }
#endif

#ifdef ISCAS_COUNT_WINDOW_GEN
    /* Windowed dense MVP input layer (dense_window=W): window size and, in ISCAS_SPARSE_VERIFY
     * builds, buffered timesteps whose batched output differed bitwise from the SDK matvec. */
    extern int iscas_debug_window_size(void);
    extern int iscas_debug_window_verify_mismatch(void);
    printf("WINDOW:size=%d:verify_mismatch=%d\n", iscas_debug_window_size(), iscas_debug_window_verify_mismatch());
#endif

#if defined(ISCAS_REPORT_RAWCYCLES) && !defined(ISCAS_COUNT_SPARSE_GEN) && ISCAS_INPUT_TEMPORAL
    /* Raw DWT inference cycles + runtime core clock for any temporal build (the sparse-MVP block
     * above already prints this line when ISCAS_COUNT_SPARSE_GEN is defined). */
    printf("RAWCYCLES:infer=%lu:clock_hz=%lu\n", (unsigned long)infer_cycles, (unsigned long)SystemCoreClock);
#endif

#ifdef ISCAS_REPORT_MEMINFO
    iscas_print_meminfo();
#endif

#ifdef ISCAS_ASYNC_PERF
    /* async_mvp builds: the MVP's RUN and STALL cycle counters summed over this inference. */
    {
        extern unsigned long iscas_debug_async_run(void);
        extern unsigned long iscas_debug_async_stall(void);
        extern unsigned long iscas_debug_async_launches(void);
        printf("ASYNCPERF:run=%lu:stall=%lu:launches=%lu\n", iscas_debug_async_run(), iscas_debug_async_stall(),
               iscas_debug_async_launches());
    }
#endif

#ifdef ISCAS_REPORT_SCORES
    /* The exact argmax input (summed final-layer scores) as IEEE-754 bit patterns. The build must
     * provide iscas_debug_scores(). */
    {
        extern const float *iscas_debug_scores(int *n);
        int n_scores = 0;
        const float *scores = iscas_debug_scores(&n_scores);
        printf("SCORES:");
        for (int i = 0; i < n_scores; i++) {
            union { float f; uint32_t u; } bits;
            bits.f = scores[i];
            printf("%s%08lx", i == 0 ? "" : ":", (unsigned long)bits.u);
        }
        printf("\n");
    }
#endif

#ifdef ISCAS_STAGE_PROFILE
    /* Per-stage DWT cycles (stage_profile=True engines), summed over all timesteps. Profiling builds
     * shift code layout slightly: use them for attribution, not for end-to-end latency. */
    extern int iscas_debug_prof_count(void);
    extern const char *iscas_debug_prof_name(int s);
    extern unsigned long iscas_debug_prof_cycles(int s);
    for (int s = 0; s < iscas_debug_prof_count(); s++) {
        printf("STAGEPROF:%s=%lu\n", iscas_debug_prof_name(s), iscas_debug_prof_cycles(s));
    }
#endif


#ifdef ISCAS_ENERGY_REPLAY
    /* Energy replay (default off; for an Advanced Energy Monitor). The request above already produced PRED and every other report line. Here the SAME inference
     * is re-run `reps` times back to back, bracketed by two EM1 gaps, so an AEM capture sees one long,
     * cleanly delimited burst whose energy / reps is the energy of one inference (the 6.25 kHz AEM
     * cannot resolve a single 40 ms inference among UART traffic). Wire: after the report lines the MCU
     * prints "REPLAY_WAIT:reps=<K>:ref_cycles=<c>"; the host then sends one command byte per capture:
     *   'R' replay the full inference K times     'U' replay only the input unpack/loop (K times; the
     *   overhead 'R' contains, to subtract)       'Q' leave replay mode (also on a 40 s command timeout).
     * Each run prints "REPLAY_DONE:<cmd>:reps=<K>:burst_cycles=<c>:clock_hz=<hz>" after the closing gap.
     * Static-input builds replay run_snn_inference() directly ('U' is then an empty loop). Temporal
     * replay needs strictly binary input frames: otherwise REPLAY_ERR. */
    {
        uint32_t ref_cycles;
#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
        ref_cycles = infer_cycles;
#else
        ref_cycles = infer_end - infer_start;
#endif
        uint32_t reps = (uint32_t)(((uint64_t)ISCAS_REPLAY_TARGET_MS * SystemCoreClock / 1000ULL) /
                                   (ref_cycles ? ref_cycles : 1U)) + 1U;
        if (reps < 2U) {
            reps = 2U;
        }
        if (reps > 500U) {
            reps = 500U;
        }
        printf("REPLAY_WAIT:reps=%lu:ref_cycles=%lu\n", (unsigned long)reps, (unsigned long)ref_cycles);

        char cmd;
        while (replay_get_cmd(&cmd) && cmd != 'Q') {
            if (cmd != 'R' && cmd != 'U') {
                continue;
            }
#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
            if (g_replay_nonbinary) {
                printf("REPLAY_ERR:nonbinary\n");
                continue;
            }
#endif
            replay_gap_ms(ISCAS_REPLAY_GAP_MS);
            uint32_t b0 = DWT->CYCCNT;
            for (uint32_t r = 0; r < reps; r++) {
#if defined(ISCAS_INPUT_TEMPORAL) && ISCAS_INPUT_TEMPORAL
                for (int t = 0; t < ISCAS_STEPS; t++) {
                    const uint8_t *row = REPLAY_ROW(t);
                    for (int i = 0; i < ISCAS_INPUT_DIM; i++) {
                        step_buf[i] = ((row[i >> 3] >> (i & 7)) & 1u) ? 1.0f : 0.0f;
                    }
                    __asm__ volatile("" ::: "memory");
                    if (cmd == 'R') {
                        if (t == 0) {
                            iscas_infer_reset(step_buf);
                        }
                        iscas_infer_step(step_buf);
                    }
                }
                if (cmd == 'R') {
                    volatile int p = iscas_infer_readout();
                    (void)p;
                }
#else
                if (cmd == 'R') {
                    volatile int p = run_snn_inference(input_buffer);
                    (void)p;
                } else {
                    __asm__ volatile("" ::: "memory");
                }
#endif
            }
            uint32_t b1 = DWT->CYCCNT;
            replay_gap_ms(ISCAS_REPLAY_GAP_MS);
            printf("REPLAY_DONE:%c:reps=%lu:burst_cycles=%lu:clock_hz=%lu\n", cmd, (unsigned long)reps,
                   (unsigned long)(b1 - b0), (unsigned long)SystemCoreClock);
        }
    }
#endif

}
