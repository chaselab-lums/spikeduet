"""Tests proving the portable-C backend FAITHFULLY EXECUTES
the Execution IR's preheader/loop schedule -- preheader ops run exactly
once, loop ops run `ISCAS_STEPS` times, and this holds whether or not
`InvariantAffineHoistingPass` ran (baseline vs. optimized pipelines
produce different generated text but IDENTICAL predictions).
"""

import pytest
import torch

from spikeduet.backend.api import compile_model
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import baseline_pipeline, optimized_pipeline
from tests.backend_c_harness import (
    CC,
    compile_portable_c,
    python_reference_predict,
    read_affine_exec_counts,
    run_harness,
    strip_c_comments,
)
from tests.test_reference import ChainSNN

pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler (gcc/clang) on PATH")


def _artifacts(dims, *, num_timesteps, input_temporal_mode, **kw):
    n = len(dims)
    betas = kw.pop("betas", [0.9] * n)
    thresholds = kw.pop("thresholds", [1.0] * n)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds, **kw)
    ir = parse_snntorch_model(model)

    base_art = compile_model(
        ir, backend="portable_c", num_timesteps=num_timesteps,
        input_temporal_mode=input_temporal_mode, pipeline=baseline_pipeline(), include_harness=True,
    )
    opt_art = compile_model(
        ir, backend="portable_c", num_timesteps=num_timesteps,
        input_temporal_mode=input_temporal_mode, pipeline=optimized_pipeline(), include_harness=True,
    )
    return ir, base_art, opt_art


# --- structural: preheader present/absent as expected -----------------------


def test_mnist_like_static_input_optimized_has_nonempty_preheader():
    """Structural check on the RAW generated text (the per-op tags like
    `/* convert0: ... */`/`/* affine0: ... */` are themselves comments and
    the intended structural markers here -- unlike the vendor-symbol scan,
    this test is not about what's CALLED, so it deliberately does not
    strip comments)."""
    _, _, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT
    )
    code = opt_art.files["portable_runtime.c"]
    assert "convert0" in code and "affine0" in code
    # the preheader marker line must appear BEFORE the "for (int t" loop
    preheader_pos = code.index("Preheader")
    loop_pos = code.index("for (int t = 0;")
    affine0_pos = code.index("affine0")
    assert preheader_pos < affine0_pos < loop_pos


def test_shd_like_temporal_input_has_empty_preheader():
    _, _, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING
    )
    code = opt_art.files["portable_runtime.c"]
    assert "<empty> -- no op was hoisted" in code


def test_baseline_pipeline_never_has_hoisted_preheader():
    _, base_art, _ = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT
    )
    code = base_art.files["portable_runtime.c"]
    assert "<empty> -- no op was hoisted" in code


def test_no_runtime_predicate_substitute_anywhere():
    """No if (t == 0) or equivalent runtime hack, in
    the actual code (comments stripped, since this check IS about what's
    executed, unlike the structural-marker tests above)."""
    _, _, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT
    )
    code = strip_c_comments(opt_art.files["portable_runtime.c"])
    assert "t == 0" not in code
    assert "t==0" not in code


# --- execution counts ----------------------------------------------


def test_mnist_like_affine0_executes_once_after_licm(tmp_path):
    ir, _, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=7, input_temporal_mode=TemporalMode.INVARIANT
    )
    counts = read_affine_exec_counts(opt_art.files, tmp_path, [0.1] * 8, num_layers=2)
    assert counts[0] == 1, f"affine0 should run exactly once (hoisted), got {counts[0]}"
    assert counts[1] == 7, f"affine1 should run every timestep (T=7), got {counts[1]}"


def test_mnist_like_affine0_executes_t_times_without_licm(tmp_path):
    ir, base_art, _ = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=7, input_temporal_mode=TemporalMode.INVARIANT
    )
    counts = read_affine_exec_counts(base_art.files, tmp_path, [0.1] * 8, num_layers=2)
    assert counts[0] == 7, f"affine0 should run every timestep without LICM, got {counts[0]}"
    assert counts[1] == 7


def test_shd_like_affine0_executes_t_times_even_after_licm(tmp_path):
    """SHD-like: temporal input means nothing is hoistable -- affine0
    still runs every timestep even when the optimized pipeline is used."""
    ir, _, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=7, input_temporal_mode=TemporalMode.VARYING
    )
    # temporal-mode harness expects T*input_dim floats -- driver here only
    # needs a same-shape buffer (values irrelevant to the count assertion)
    counts = read_affine_exec_counts(opt_art.files, tmp_path, [0.1] * (8 * 7), num_layers=2)
    assert counts[0] == 7
    assert counts[1] == 7


def test_deep_chain_only_layer0_ever_hoisted(tmp_path):
    """Arbitrary depth: LICM must never hoist anything past layer 0 in
    this single-chain topology (every later layer depends on a spike
    buffer, never provably invariant)."""
    ir, _, opt_art = _artifacts(
        [(6, 8), (8, 6), (6, 5), (5, 3)], num_timesteps=6, input_temporal_mode=TemporalMode.INVARIANT
    )
    counts = read_affine_exec_counts(opt_art.files, tmp_path, [0.1] * 6, num_layers=4)
    assert counts[0] == 1
    assert counts[1:] == [6, 6, 6]


# --- prediction equivalence --------------------------------------


def test_optimized_and_baseline_predictions_agree_static(tmp_path):
    ir, base_art, opt_art = _artifacts(
        [(8, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT
    )
    base_exe = compile_portable_c(base_art.files, tmp_path / "base")
    opt_exe = compile_portable_c(opt_art.files, tmp_path / "opt")

    torch.manual_seed(11)
    samples = [torch.randn(1, 8) for _ in range(12)]
    flat = [x[0].tolist() for x in samples]

    base_pred = run_harness(base_exe, flat)
    opt_pred = run_harness(opt_exe, flat)
    expected = [python_reference_predict(ir, [x] * 5) for x in samples]

    assert base_pred == expected
    assert opt_pred == expected
    assert base_pred == opt_pred


def test_optimized_and_baseline_predictions_agree_temporal(tmp_path):
    ir, base_art, opt_art = _artifacts(
        [(4, 6), (6, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING
    )
    base_exe = compile_portable_c(base_art.files, tmp_path / "base")
    opt_exe = compile_portable_c(opt_art.files, tmp_path / "opt")

    torch.manual_seed(12)
    sequences = [[torch.randn(1, 4) for _ in range(5)] for _ in range(10)]
    flat = []
    for seq in sequences:
        row = []
        for x in seq:
            row.extend(x[0].tolist())
        flat.append(row)

    base_pred = run_harness(base_exe, flat)
    opt_pred = run_harness(opt_exe, flat)
    expected = [python_reference_predict(ir, seq) for seq in sequences]

    assert base_pred == expected
    assert opt_pred == expected
    assert base_pred == opt_pred


def test_optimized_matches_baseline_with_full_l1_l5_composition(tmp_path):
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    ir, base_art, opt_art = _artifacts(
        [(4, 4), (4, 3)], num_timesteps=6, input_temporal_mode=TemporalMode.INVARIANT,
        betas=[[0.0, 0.4, 0.8, 1.0], 0.9],
        thresholds=[[0.3, 0.6, 0.9, 1.2], 1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
        reset_mechanisms=["zero", "subtract"],
        reset_delays=[True, False],
        inhibitions=[True, False],
        state_quants=[q, False],
    )
    base_exe = compile_portable_c(base_art.files, tmp_path / "base")
    opt_exe = compile_portable_c(opt_art.files, tmp_path / "opt")

    torch.manual_seed(13)
    samples = [torch.randn(1, 4) for _ in range(10)]
    flat = [x[0].tolist() for x in samples]

    base_pred = run_harness(base_exe, flat)
    opt_pred = run_harness(opt_exe, flat)
    expected = [python_reference_predict(ir, [x] * 6) for x in samples]

    assert base_pred == expected
    assert opt_pred == expected
