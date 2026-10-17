/* snn.h -- engine contract and shared LIF helpers for SpikeDuet EFR32 builds.
 *
 * Every generated engine (snn.c) includes this header and calls the helpers below, so the neuron
 * semantics have exactly one C implementation on the device. They mirror the reference simulator
 * (spikeduet/reference/simulator.py) and snntorch.Leaky 1.0.0, and the portable-C backend carries
 * an identical copy; keep the two in sync.
 *
 * iscas_lif_update_inhibition_layer() reads iscas_layer_t, which the generated model_data.h
 * defines: include model_data.h before this header.
 */
#ifndef SNN_H
#define SNN_H

#include <stdio.h>
#include <math.h>

/* Wraps every malloc/calloc in the engines: on failure it prints ALLOC_FAIL:<what> over the same
 * UART as the PRED replies and halts instead of running with a missing buffer. */
static inline void *snn_alloc_check(void *ptr, const char *what)
{
    if (ptr == NULL) {
        printf("ALLOC_FAIL:%s\n", what);
        while (1) {
            /* halt visibly rather than run with a missing buffer */
        }
    }
    return ptr;
}

/* Per-neuron parameter lookup: len == 1 is one value shared by all neurons, otherwise arr[j]. */
static inline float iscas_param_at(const float *arr, int len, int j)
{
    return (len == 1) ? arr[0] : arr[j];
}

/* Nearest level; on an exact tie the first (lowest, levels are ascending) level wins, as in
 * snntorch.functional.quant.state_quant. The strict < keeps the first of equal distances. */
static inline float iscas_state_quantize(float x, const float *levels, int n_levels)
{
    float best = levels[0];
    float best_diff = fabsf(levels[0] - x);
    for (int i = 1; i < n_levels; i++) {
        float diff = fabsf(levels[i] - x);
        if (diff < best_diff) {
            best_diff = diff;
            best = levels[i];
        }
    }
    return best;
}

/* One neuron, one timestep. reset_mechanism: 0 = subtract, 1 = zero, 2 = none (snnTorch's
 * reset_dict numbering); reset_delay: 1 applies the reset on the next step. With
 * r = (old_mem > threshold):
 *
 *   candidate = beta*old_mem + current - r*threshold     subtract
 *             = beta*(1 - r)*old_mem + current           zero
 *             = beta*old_mem + current                   none
 *   candidate = Q(candidate) if the layer quantizes state
 *   spike     = candidate > threshold
 *   immediate reset: d = spike - r, new_mem = candidate - d*threshold (subtract)
 *                                  new_mem = (1 - d)*candidate      (zero)
 *
 * graded_spikes_factor scales only the emitted spike, never the reset arithmetic. The branches
 * are kept explicit so they can be checked line by line against the reference simulator. */
static inline void iscas_lif_update(
    float current, float old_mem, float beta, float threshold,
    int reset_mechanism, int reset_delay, float graded_spikes_factor,
    const float *state_quant_levels, int state_quant_num_levels,
    float *out_spike, float *out_new_mem)
{
    float reset_old = (old_mem > threshold) ? 1.0f : 0.0f;
    float spike, new_mem;

    if (reset_mechanism == 0) {           /* subtract */
        if (!reset_delay) {
            float updated = (beta * old_mem) + current - reset_old * threshold;
            if (state_quant_num_levels > 0) {
                updated = iscas_state_quantize(updated, state_quant_levels, state_quant_num_levels);
            }
            spike = (updated > threshold) ? 1.0f : 0.0f;
            float do_reset = spike - reset_old;
            new_mem = updated - do_reset * threshold;
        } else {
            new_mem = (beta * old_mem) + current - reset_old * threshold;
            if (state_quant_num_levels > 0) {
                new_mem = iscas_state_quantize(new_mem, state_quant_levels, state_quant_num_levels);
            }
            spike = (new_mem > threshold) ? 1.0f : 0.0f;
        }
    } else if (reset_mechanism == 1) {    /* zero */
        float zeroed = (1.0f - reset_old) * old_mem;
        if (!reset_delay) {
            float updated = (beta * zeroed) + current;
            if (state_quant_num_levels > 0) {
                updated = iscas_state_quantize(updated, state_quant_levels, state_quant_num_levels);
            }
            spike = (updated > threshold) ? 1.0f : 0.0f;
            float do_reset = spike - reset_old;
            new_mem = (1.0f - do_reset) * updated;
        } else {
            new_mem = (beta * zeroed) + current;
            if (state_quant_num_levels > 0) {
                new_mem = iscas_state_quantize(new_mem, state_quant_levels, state_quant_num_levels);
            }
            spike = (new_mem > threshold) ? 1.0f : 0.0f;
        }
    } else {                               /* none: reset_delay has no effect */
        new_mem = (beta * old_mem) + current;
        if (state_quant_num_levels > 0) {
            new_mem = iscas_state_quantize(new_mem, state_quant_levels, state_quant_num_levels);
        }
        spike = (new_mem > threshold) ? 1.0f : 0.0f;
    }

    *out_spike = spike * graded_spikes_factor;
    *out_new_mem = new_mem;
}

/* A whole layer with winner-take-all inhibition: only the neuron with the largest
 * candidate - threshold margin (lowest index on a tie) may spike. Two snnTorch 1.0.0 quirks are
 * reproduced exactly: the emitted spike is never scaled by graded_spikes_factor, and the
 * immediate-reset correction uses spike / graded_spikes_factor. mem_state_scratch and
 * reset_old_scratch are caller-provided float[out_features] buffers. */
static inline void iscas_lif_update_inhibition_layer(
    const iscas_layer_t *L, const float *current, float *mem,
    float *mem_state_scratch, float *reset_old_scratch,
    float *out_spike)
{
    int n = L->out_features;
    int winner = 0;
    float best_margin = 0.0f;

    /* Pass 1: candidate membranes and the winner. */
    for (int j = 0; j < n; j++) {
        float beta = iscas_param_at(L->beta, L->beta_len, j);
        float threshold = iscas_param_at(L->threshold, L->threshold_len, j);
        float old_mem = mem[j];
        float reset_old = (old_mem > threshold) ? 1.0f : 0.0f;
        float mem_state;

        if (L->reset_mechanism == 0) {          /* subtract */
            mem_state = (beta * old_mem) + current[j] - reset_old * threshold;
        } else if (L->reset_mechanism == 1) {   /* zero */
            float zeroed = (1.0f - reset_old) * old_mem;
            mem_state = (beta * zeroed) + current[j];
        } else {                                 /* none */
            mem_state = (beta * old_mem) + current[j];
        }

        if (L->state_quant_num_levels > 0) {
            mem_state = iscas_state_quantize(mem_state, L->state_quant_levels, L->state_quant_num_levels);
        }

        mem_state_scratch[j] = mem_state;
        reset_old_scratch[j] = reset_old;

        float margin = mem_state - threshold;
        if (j == 0 || margin > best_margin) {
            best_margin = margin;
            winner = j;
        }
    }

    /* Pass 2: winner-masked spikes and, for an immediate reset, the correction. */
    for (int j = 0; j < n; j++) {
        float threshold = iscas_param_at(L->threshold, L->threshold_len, j);
        float mem_state = mem_state_scratch[j];
        float reset_old = reset_old_scratch[j];
        float margin = mem_state - threshold;
        float fired = (margin > 0.0f) ? 1.0f : 0.0f;
        float spike = (j == winner) ? fired : 0.0f;  /* never scaled by the graded factor */
        float new_mem = mem_state;

        if (!L->reset_delay && (L->reset_mechanism == 0 || L->reset_mechanism == 1)) {
            float graded_factor = iscas_param_at(L->graded_spikes_factor, L->graded_spikes_factor_len, j);
            float do_reset = (spike / graded_factor) - reset_old;
            if (L->reset_mechanism == 0) {       /* subtract */
                new_mem = mem_state - do_reset * threshold;
            } else {                              /* zero */
                new_mem = mem_state - do_reset * mem_state;
            }
        }

        mem[j] = new_mem;
        out_spike[j] = spike;
    }
}

/* Engine contract, implemented by the generated snn.c. */
void snn_init(void);

/* One inference over ISCAS_STEPS steps; returns the argmax of the summed final-layer membranes.
 * input_data is [ISCAS_INPUT_DIM] (reused every step) or [ISCAS_STEPS][ISCAS_INPUT_DIM] when
 * ISCAS_INPUT_TEMPORAL is set. */
int run_snn_inference(float *input_data);

/* The same inference, streamed one input vector per step (app.c uses this for temporal models so
 * the whole input never has to be resident): reset, ISCAS_STEPS x step, readout. */
void iscas_infer_reset(const float *first_input);
void iscas_infer_step(const float *step_input);
int iscas_infer_readout(void);

#endif /* SNN_H */
