"""Structural tests of the generated EFR32 CPU and MVP engines.

They check the generated text: the preheader/loop split, that hoisting needs no runtime predicate,
MVP call-site counts, fusion and the direct-FP16 spike writes. Host tests that compile and run
EFR32 code are in test_efr32_streaming.py and test_efr32_cpu_sparse.py.
"""

from spikeduet.backend.efr32.codegen import generate_efr32_cpu_schedule, generate_efr32_mvp_schedule
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import baseline_pipeline, compile_source_to_execution_ir, optimized_pipeline
from tests.backend_c_harness import strip_c_comments
from tests.test_reference import ChainSNN


def _module(dims, *, num_timesteps, input_temporal_mode, pipeline):
    n = len(dims)
    model = ChainSNN(dims=list(dims), betas=[0.9] * n, thresholds=[1.0] * n)
    ir = parse_snntorch_model(model)
    return compile_source_to_execution_ir(
        ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode, pipeline=pipeline
    )


def test_cpu_schedule_mnist_like_has_hoisted_preheader():
    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    code = generate_efr32_cpu_schedule(mod)
    preheader_pos = code.index("Preheader")
    affine0_pos = code.index("affine0")
    loop_pos = code.index("void iscas_infer_step(")
    assert preheader_pos < affine0_pos < loop_pos


def test_cpu_schedule_shd_like_has_empty_preheader():
    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING,
        pipeline=optimized_pipeline(),
    )
    code = generate_efr32_cpu_schedule(mod)
    assert "<empty> -- no op was hoisted" in code


def test_cpu_schedule_has_no_runtime_hoisting_predicate():
    """Hoisting is structural: the generated code never tests the timestep index."""
    for pipeline in (baseline_pipeline(), optimized_pipeline()):
        mod = _module(
            [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
            pipeline=pipeline,
        )
        code = strip_c_comments(generate_efr32_cpu_schedule(mod))
        assert "t == 0" not in code and "t==0" not in code


def test_cpu_schedule_reuses_shared_snn_h_helpers_not_reimplemented():
    """The generated CPU file must #include snn.h and call its helpers --
    never redefine iscas_lif_update()/iscas_param_at() itself (unlike the
    portable-C backend, which deliberately DOES reimplement them to stay
    independent of runtime/efr32/ -- see that backend's own docstring)."""
    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    code = generate_efr32_cpu_schedule(mod)
    assert '#include "snn.h"' in code
    assert "static inline void iscas_lif_update(" not in code
    assert "iscas_lif_update(" in code  # called, not redefined


def test_cpu_schedule_uses_snn_alloc_check():
    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    code = generate_efr32_cpu_schedule(mod)
    assert "snn_alloc_check(" in code


def test_cpu_schedule_public_contract_matches_hand_written_engine():
    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    code = generate_efr32_cpu_schedule(mod)
    assert "void snn_init(void)" in code
    assert "int run_snn_inference(float *input_data)" in code


def test_mvp_schedule_hoisted_affine_has_single_call_site_outside_loop():
    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    raw = generate_efr32_mvp_schedule(mod)
    stripped = strip_c_comments(raw)
    call_sites = stripped.count("sl_math_mvp_matrix_vector_mult_f16(")
    # exactly one call site per layer (2 layers) -- structural, not
    # runtime-predicated (a naive "always call inside loop + skip" design
    # would still show exactly one call SITE in source text too, so the
    # decisive check is the call site's TEXTUAL position, below).
    assert call_sites == 2

    preheader_pos = raw.index("Preheader")
    loop_pos = raw.index("void iscas_infer_step(")
    # search AFTER the preheader marker -- the file's own header doc
    # comment (before run_snn_inference even starts) also mentions this
    # function name in prose, which would otherwise be found first.
    first_call_pos = raw.index("sl_math_mvp_matrix_vector_mult_f16(", preheader_pos)
    # the hoisted layer's call site must be BEFORE the loop
    assert preheader_pos < first_call_pos < loop_pos


def test_mvp_schedule_shd_like_call_site_inside_loop_only():
    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING,
        pipeline=optimized_pipeline(),
    )
    raw = generate_efr32_mvp_schedule(mod)
    assert "<empty> -- no op was hoisted" in raw
    preheader_pos = raw.index("Preheader")
    loop_pos = raw.index("void iscas_infer_step(")
    preheader_region = raw[preheader_pos:loop_pos]
    assert "sl_math_mvp_matrix_vector_mult_f16(" not in preheader_region
    # the (only) real call site is inside the loop
    assert "sl_math_mvp_matrix_vector_mult_f16(" in raw[loop_pos:]


def test_mvp_schedule_has_no_runtime_hoisting_predicate():
    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    assert "t == 0" not in code and "t==0" not in code


def test_mvp_schedule_converts_weights_to_fp16_at_init():
    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    code = generate_efr32_mvp_schedule(mod)
    assert "float16_t" in code
    assert "sl_math_matrix_init_f16(" in code


def test_efr32_backend_default_unchanged_no_engine_kwarg():
    """Without an engine the backend emits model_data.h only."""
    from spikeduet.backend.efr32.codegen import Efr32Backend

    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    artifact = Efr32Backend().generate(mod)
    assert set(artifact.files) == {"model_data.h"}


def test_efr32_backend_engine_cpu_adds_schedule_file():
    from spikeduet.backend.efr32.codegen import Efr32Backend

    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    artifact = Efr32Backend().generate(mod, engine="cpu")
    assert set(artifact.files) == {"model_data.h", "snn_generated_cpu.c"}


def test_efr32_backend_engine_mvp_adds_schedule_file():
    from spikeduet.backend.efr32.codegen import Efr32Backend

    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    artifact = Efr32Backend().generate(mod, engine="mvp")
    assert set(artifact.files) == {"model_data.h", "snn_generated_mvp.c"}


def test_efr32_backend_unknown_engine_raises():
    import pytest

    from spikeduet.backend.efr32.codegen import Efr32Backend

    mod = _module([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline())
    with pytest.raises(ValueError, match="unknown engine"):
        Efr32Backend().generate(mod, engine="dsp")


def test_deep_chain_cpu_schedule_only_layer0_hoisted():
    mod = _module(
        [(6, 8), (8, 6), (6, 5), (5, 3)], num_timesteps=6, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    code = generate_efr32_cpu_schedule(mod)
    preheader_block = code[code.index("Preheader"):code.index("void iscas_infer_step(")]
    assert "affine0" in preheader_block
    for later in ("affine1", "affine2", "affine3"):
        assert later not in preheader_block


# --- spike-staging fusion in the generated EFR32 engines ---


def test_cpu_schedule_fusion_removes_intermediate_convert():
    from spikeduet.ir.pass_manager import fusion_pipeline

    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=fusion_pipeline(),
    )
    code = generate_efr32_cpu_schedule(mod)
    assert "convert1" not in code and "convert2" not in code
    assert "convert0" in code  # layer 0's own input staging is never fused
    assert "-> stage1" in code  # LIF now writes the downstream stage buffer directly
    assert "-> stage2" in code


def test_mvp_schedule_fusion_removes_intermediate_convert_fp16_never_dropped():
    """Fusion removes the FP32 staging copy on fused edges; the FP16 conversion itself must still
    happen for every layer, either in a staging loop or as a direct write from the upstream LIF."""
    from spikeduet.ir.pass_manager import fusion_pipeline

    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=fusion_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    assert "convert1" not in code and "convert2" not in code
    assert code.count("sl_math_mvp_matrix_vector_mult_f16(") == 3  # every layer still MVP-realized

    import re

    staging_casts = re.findall(r"g_in_f16\[\d+\]\[i\] = \(float16_t\)", code)
    direct_writes = re.findall(r"g_in_f16\[\d+\]\[j\] = \(float16_t\)spike;", code)
    # every layer's FP16 input is populated by EXACTLY one of the two
    # paths -- never both, never neither.
    assert len(staging_casts) + len(direct_writes) == 3


def test_mvp_schedule_direct_fp16_on_fused_noninhibited_edges():
    """A compiler-proven-fused, non-inhibited edge writes
    its final spike directly into g_in_f16 (no separate staging loop);
    layer 0 (fed by the raw input, never a fusion target) keeps the
    staged path regardless of pipeline."""
    from spikeduet.ir.pass_manager import fusion_pipeline

    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=fusion_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))

    import re

    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    assert staging_casts == ["0"]  # only layer 0 (no upstream LIF) stays staged
    assert sorted(direct_writes) == ["1", "2"]  # layers 1 and 2 (fed by lif0/lif1) go direct


def test_licm_plus_fusion_cpu_schedule_matches_pretty_printed_expectation():
    """Static input, LICM + fusion: preheader has convert0+affine0; loop has
    lif0 (writing stage1 directly) -> affine1 -> lif1 (writing stage2) ->
    affine2 -> lif2 (final, unfused)."""
    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    code = generate_efr32_cpu_schedule(mod)
    preheader_block = code[code.index("Preheader"):code.index("void iscas_infer_step(")]
    loop_block = code[code.index("void iscas_infer_step("):]

    assert "convert0" in preheader_block and "affine0" in preheader_block
    assert "convert1" not in loop_block and "convert2" not in loop_block
    assert "lif0" in loop_block and "-> stage1" in loop_block
    assert "lif1" in loop_block and "-> stage2" in loop_block
    assert "lif2" in loop_block and "-> spike2" in loop_block  # final output, unfused


def test_temporal_mvp_schedule_fusion_with_empty_preheader():
    from spikeduet.ir.pass_manager import optimized_pipeline as _opt

    mod = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING,
        pipeline=_opt(),
    )
    code = generate_efr32_mvp_schedule(mod)
    assert "<empty> -- no op was hoisted" in code
    assert "convert1" not in strip_c_comments(code)
    assert "convert0" in code  # a varying input layer's staging is neither hoisted nor fused


# --- direct-FP16 spike writes on fused MVP edges ---


def test_mvp_schedule_baseline_pipeline_retains_old_staged_path():
    """baseline_pipeline() runs neither LICM nor fusion -- the MVP
    realization must stage every layer: zero direct-FP16 writes anywhere, every layer's FP16 input
    populated by its own staging loop."""
    import re

    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=baseline_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    assert direct_writes == []
    assert sorted(staging_casts) == ["0", "1", "2"]
    assert "g_mvp_direct_fp16_write_count" in code
    assert "g_mvp_stage_cast_count" in code


def test_mvp_schedule_licm_pipeline_retains_old_staged_path():
    """licm_pipeline() (hoisting only, no fusion) must also keep every
    layer on the old staged path -- direct-FP16 materialization is gated
    strictly on a proven-fused edge, which only SpikeStagingFusionPass
    (via fusion_pipeline()/optimized_pipeline()) can produce."""
    import re

    from spikeduet.ir.pass_manager import licm_pipeline

    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=licm_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    assert direct_writes == []
    assert sorted(staging_casts) == ["0", "1", "2"]


def test_mvp_schedule_inhibited_edge_stays_on_staged_path():
    """An inhibited LIF layer's output, even when fusion-eligible by the
    ConvertOp/AffineOp dataflow shape, must NOT be direct-FP16-materialized
    (see `mvp_emitters.mvp_direct_fused_layer_indices`)."""
    import re

    from spikeduet.ir.pass_manager import fusion_pipeline

    model = ChainSNN(
        dims=[(8, 6), (6, 5), (5, 3)], betas=[0.9, 0.9, 0.9], thresholds=[1.0, 1.0, 1.0],
        inhibitions=[True, False, False],
    )
    ir = parse_snntorch_model(model)
    mod = compile_source_to_execution_ir(
        ir, num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=fusion_pipeline(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    # layer 1 is fed by the INHIBITED lif0 -> must stay staged, not direct
    assert "1" in staging_casts
    assert "1" not in direct_writes
    # layer 2 is fed by the non-inhibited lif1 -> still goes direct
    assert "2" in direct_writes



def test_mvp_schedule_three_layers_two_direct_fused_edges():
    """784->64->32->10, static input:
    layer 0 is fed by the raw input (never fusable); layers 1 and 2 are
    each fed by a non-inhibited LIF -> both direct-FP16, exactly 2 fused
    hidden edges."""
    import re

    from spikeduet.ir.pass_manager import optimized_pipeline as _opt

    mod = _module(
        [(784, 64), (64, 32), (32, 10)], num_timesteps=10, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=_opt(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    assert sorted(direct_writes) == ["1", "2"]
    assert staging_casts == ["0"]


def test_mvp_schedule_temporal_two_layers_one_direct_fused_edge():
    """700->64->20, time-varying input: layer 0 is fed by the per-timestep input (never fusable, and
    never LICM-hoistable either since it's VARYING); layer 1 is fed by a
    non-inhibited LIF -> direct-FP16. Exactly 1 fused hidden edge, proving
    the optimization still applies despite temporal input."""
    import re

    from spikeduet.ir.pass_manager import optimized_pipeline as _opt

    mod = _module(
        [(700, 64), (64, 20)], num_timesteps=100, input_temporal_mode=TemporalMode.VARYING,
        pipeline=_opt(),
    )
    code = strip_c_comments(generate_efr32_mvp_schedule(mod))
    direct_writes = re.findall(r"g_in_f16\[(\d+)\]\[j\] = \(float16_t\)spike;", code)
    staging_casts = re.findall(r"g_in_f16\[(\d+)\]\[i\] = \(float16_t\)", code)
    assert direct_writes == ["1"]
    assert staging_casts == ["0"]


def test_cpu_and_mvp_schedule_define_iscas_input_temporal():
    """Both engines define ISCAS_INPUT_TEMPORAL, which their input staging reads."""
    mod_static = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=baseline_pipeline(),
    )
    mod_temporal = _module(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING,
        pipeline=baseline_pipeline(),
    )
    assert "#define ISCAS_INPUT_TEMPORAL 0" in generate_efr32_cpu_schedule(mod_static)
    assert "#define ISCAS_INPUT_TEMPORAL 1" in generate_efr32_cpu_schedule(mod_temporal)
    assert "#define ISCAS_INPUT_TEMPORAL 0" in generate_efr32_mvp_schedule(mod_static)
    assert "#define ISCAS_INPUT_TEMPORAL 1" in generate_efr32_mvp_schedule(mod_temporal)


def test_mvp_schedule_direct_fp16_counters_declared_only_in_mvp():
    """The direct-FP16 debug counters are MVP-only -- the
    CPU schedule must never declare or reference them."""
    mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=optimized_pipeline(),
    )
    cpu_code = generate_efr32_cpu_schedule(mod)
    assert "g_mvp_direct_fp16_write_count" not in cpu_code
    assert "g_mvp_stage_cast_count" not in cpu_code
    from spikeduet.ir.pass_manager import fusion_pipeline

    mvp_mod = _module(
        [(8, 6), (6, 5), (5, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
        pipeline=fusion_pipeline(),
    )
    mvp_code = generate_efr32_mvp_schedule(mvp_mod)
    assert "iscas_debug_mvp_stage_cast_count" in mvp_code
    assert "iscas_debug_mvp_direct_fp16_write_count" in mvp_code
