"""Host-compiled conformance tests for the portable-C
backend's realization of `SpikeStagingFusionPass` -- proves the fused
generated C actually compiles, runs, and produces numerically identical
predictions to the unfused baseline, across every all Leaky semantic
combination (graded spikes, inhibition, state quantization) that a fusion
bug could plausibly corrupt if it touched semantics instead of only
renaming a write target.
"""

import pytest
import torch

from spikeduet.backend.api import compile_model
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import baseline_pipeline, fusion_pipeline, licm_pipeline, optimized_pipeline
from tests.backend_c_harness import (
    CC,
    compile_portable_c,
    python_reference_predict,
    read_exec_counts,
    run_harness,
)
from tests.test_reference import ChainSNN

pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler (gcc/clang) on PATH")


def _artifacts(dims, *, num_timesteps, input_temporal_mode=TemporalMode.INVARIANT, **kw):
    n = len(dims)
    betas = kw.pop("betas", [0.9] * n)
    thresholds = kw.pop("thresholds", [1.0] * n)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds, **kw)
    ir = parse_snntorch_model(model)

    base_art = compile_model(
        ir, backend="portable_c", num_timesteps=num_timesteps,
        input_temporal_mode=input_temporal_mode, pipeline=baseline_pipeline(), include_harness=True,
    )
    fused_art = compile_model(
        ir, backend="portable_c", num_timesteps=num_timesteps,
        input_temporal_mode=input_temporal_mode, pipeline=fusion_pipeline(), include_harness=True,
    )
    return ir, base_art, fused_art


def _compare(tmp_path, ir, base_art, fused_art, samples, num_timesteps):
    base_exe = compile_portable_c(base_art.files, tmp_path / "base")
    fused_exe = compile_portable_c(fused_art.files, tmp_path / "fused")

    flat = [x[0].tolist() for x in samples]
    base_pred = run_harness(base_exe, flat)
    fused_pred = run_harness(fused_exe, flat)
    expected = [python_reference_predict(ir, [x] * num_timesteps) for x in samples]

    assert base_pred == expected
    assert fused_pred == expected
    assert base_pred == fused_pred


# --- structural: fused code has no intermediate copy -------------------------


def test_fused_code_has_no_intermediate_convert_call():
    _, base_art, fused_art = _artifacts([(8, 6), (6, 5), (5, 3)], num_timesteps=5)
    base_code = base_art.files["portable_runtime.c"]
    fused_code = fused_art.files["portable_runtime.c"]

    assert "convert1" in base_code and "convert2" in base_code
    assert "convert1" not in fused_code and "convert2" not in fused_code
    assert "convert0" in fused_code  # layer 0's own staging is untouched


# --- graded spikes (non-binary), scalar and per-neuron ------------------------


def test_fused_matches_unfused_graded_scalar(tmp_path):
    ir, base_art, fused_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6, graded_spikes_factors=[2.5, 1.0]
    )
    torch.manual_seed(20)
    samples = [torch.randn(1, 4) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


def test_fused_matches_unfused_graded_per_neuron(tmp_path):
    ir, base_art, fused_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6,
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
    )
    torch.manual_seed(21)
    samples = [torch.randn(1, 4) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


# --- inhibition ---------------------------------------------------------------


def test_fused_matches_unfused_inhibition(tmp_path):
    ir, base_art, fused_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6, inhibitions=[True, False]
    )
    torch.manual_seed(22)
    samples = [torch.randn(1, 4) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


# --- state quantization --------------------------------------------------------


def test_fused_matches_unfused_state_quant(tmp_path):
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    ir, base_art, fused_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6, state_quants=[q, False]
    )
    torch.manual_seed(23)
    samples = [torch.randn(1, 4) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


# --- full graded + inhibition + state-quant composition -------------------------------------------------


def test_fused_matches_unfused_full_composition(tmp_path):
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    ir, base_art, fused_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6,
        betas=[[0.0, 0.4, 0.8, 1.0], 0.9],
        thresholds=[[0.3, 0.6, 0.9, 1.2], 1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
        reset_mechanisms=["zero", "subtract"],
        reset_delays=[True, False],
        inhibitions=[True, False],
        state_quants=[q, False],
    )
    torch.manual_seed(24)
    samples = [torch.randn(1, 4) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


# --- arbitrary depth (3+ intermediate boundaries) ------------------------------


def test_fused_matches_unfused_arbitrary_depth(tmp_path):
    ir, base_art, fused_art = _artifacts(
        [(6, 8), (8, 6), (6, 5), (5, 3)], num_timesteps=6
    )
    fused_code = fused_art.files["portable_runtime.c"]
    for i in (1, 2, 3):
        assert f"convert{i}" not in fused_code
    torch.manual_seed(25)
    samples = [torch.randn(1, 6) for _ in range(10)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 6)


# --- final output layer: no inappropriate elimination -------------------------


def test_final_layer_spike_buffer_still_readable_after_fusion(tmp_path):
    """The final layer's spike buffer must remain intact and correct --
    it's both the readout source (argmax input) AND, if
    ISCAS_PORTABLE_EXPOSE_MEMSUM were built, a debug source. Proven here
    via matching predictions (which are computed from the final layer's
    membrane, itself driven by whether spikes fed back correctly) across
    the WHOLE pipeline, not just the intermediate fused edges."""
    ir, base_art, fused_art = _artifacts([(4, 6), (6, 3)], num_timesteps=8)
    torch.manual_seed(26)
    samples = [torch.randn(1, 4) for _ in range(15)]
    _compare(tmp_path, ir, base_art, fused_art, samples, 8)


# --- copy execution counters -----------------------


def test_copy_counts_across_all_four_pipelines_mnist_like(tmp_path):
    """MNIST-like (static input), 3 layers -- two intermediate staging
    boundaries (convert1, convert2). Expected :
      baseline:       affine0=T, convert1=T, convert2=T
      LICM only:      affine0=1, convert1=T, convert2=T
      fusion only:    affine0=T, convert1=0, convert2=0
      LICM+fusion:    affine0=1, convert1=0, convert2=0
    """
    dims = [(8, 6), (6, 5), (5, 3)]
    model = ChainSNN(dims=dims, betas=[0.9] * 3, thresholds=[1.0] * 3)
    ir = parse_snntorch_model(model)
    T = 5
    x = [0.1] * 8

    # convert0 (layer 0's own staging, from the raw input) is NEVER a
    # fusion target (no upstream LIF feeds it) -- but IS a LICM hoisting
    # target (it feeds affine0, which IS hoisted): T under baseline/
    # fusion-only, 1 under licm/optimized (hoisted into the preheader,
    # same as affine0).
    for name, pipeline, expected_affine0, expected_convert0, expected_convert in [
        ("baseline", baseline_pipeline(), T, T, T),
        ("licm", licm_pipeline(), 1, 1, T),
        ("fusion", fusion_pipeline(), T, T, 0),
        ("optimized", optimized_pipeline(), 1, 1, 0),
    ]:
        artifact = compile_model(
            ir, backend="portable_c", num_timesteps=T,
            input_temporal_mode=TemporalMode.INVARIANT, pipeline=pipeline,
        )
        affine_counts = read_exec_counts(artifact.files, tmp_path / f"{name}_a", x, 3, "affine")
        convert_counts = read_exec_counts(artifact.files, tmp_path / f"{name}_c", x, 3, "convert")

        assert affine_counts[0] == expected_affine0, f"{name}: affine0={affine_counts[0]}, want {expected_affine0}"
        assert affine_counts[1] == T and affine_counts[2] == T, f"{name}: downstream affines should always run every timestep"
        assert convert_counts[0] == expected_convert0, f"{name}: convert0={convert_counts[0]}, want {expected_convert0}"
        assert convert_counts[1] == expected_convert, f"{name}: convert1={convert_counts[1]}, want {expected_convert}"
        assert convert_counts[2] == expected_convert, f"{name}: convert2={convert_counts[2]}, want {expected_convert}"


def test_copy_counts_shd_like_temporal(tmp_path):
    """SHD-like (temporal input), 2 layers -- one intermediate staging
    boundary (convert1). LICM finds nothing hoistable (affine0=T in every
    pipeline); fusion still removes convert1's runtime executions to 0."""
    dims = [(8, 6), (6, 3)]
    model = ChainSNN(dims=dims, betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)
    T = 5
    x = [0.1] * (8 * T)

    for name, pipeline, expected_convert1 in [
        ("baseline", baseline_pipeline(), T),
        ("licm", licm_pipeline(), T),
        ("fusion", fusion_pipeline(), 0),
        ("optimized", optimized_pipeline(), 0),
    ]:
        artifact = compile_model(
            ir, backend="portable_c", num_timesteps=T,
            input_temporal_mode=TemporalMode.VARYING, pipeline=pipeline,
        )
        affine_counts = read_exec_counts(artifact.files, tmp_path / f"{name}_a", x, 2, "affine")
        convert_counts = read_exec_counts(artifact.files, tmp_path / f"{name}_c", x, 2, "convert")

        assert affine_counts[0] == T, f"{name}: affine0 must run every timestep under temporal input"
        assert convert_counts[0] == T  # layer 0's own staging, never fused
        assert convert_counts[1] == expected_convert1, f"{name}: convert1={convert_counts[1]}, want {expected_convert1}"


# --- full optimized (LICM + fusion) end-to-end ---------------------------------


def test_licm_plus_fusion_matches_baseline(tmp_path):
    model = ChainSNN(
        dims=[(8, 6), (6, 5), (5, 3)], betas=[0.9, 0.9, 0.9], thresholds=[1.0, 1.0, 1.0]
    )
    ir = parse_snntorch_model(model)

    base_art = compile_model(
        ir, backend="portable_c", num_timesteps=5,
        input_temporal_mode=TemporalMode.INVARIANT, pipeline=baseline_pipeline(), include_harness=True,
    )
    both_art = compile_model(
        ir, backend="portable_c", num_timesteps=5,
        input_temporal_mode=TemporalMode.INVARIANT, pipeline=optimized_pipeline(), include_harness=True,
    )
    both_code = both_art.files["portable_runtime.c"]
    assert "convert1" not in both_code and "convert2" not in both_code
    preheader_pos = both_code.index("Preheader")
    loop_pos = both_code.index("for (int t = 0;")
    assert preheader_pos < both_code.index("affine0") < loop_pos

    base_exe = compile_portable_c(base_art.files, tmp_path / "base")
    both_exe = compile_portable_c(both_art.files, tmp_path / "both")

    torch.manual_seed(27)
    samples = [torch.randn(1, 8) for _ in range(10)]
    flat = [x[0].tolist() for x in samples]
    base_pred = run_harness(base_exe, flat)
    both_pred = run_harness(both_exe, flat)
    expected = [python_reference_predict(ir, [x] * 5) for x in samples]

    assert base_pred == expected
    assert both_pred == expected
