"""Range-sparse MVP realization (spikeduet/backend/efr32/codegen.py, opt-in `sparse_mvp=True`).

Structural/textual tests only: the realization programs MVP registers directly, so its
numerical correctness is established on EFR32MG24 silicon by the ISCAS_SPARSE_VERIFY
build (bitwise comparison against the deployed dense SDK kernel on every timestep), not
here. These tests pin down the eligibility policy, the fallback, and that the default
output is untouched.
"""
import re

import pytest

from spikeduet.backend.efr32.codegen import Efr32Backend, generate_efr32_mvp_schedule
from spikeduet.backend.efr32.cost_model import sparse_merge_gap_pairs
from spikeduet.backend.efr32.mvp_emitters import mvp_sparse_plan
from spikeduet.backend.efr32.mvp_sparse import EFR32_MVP_LAUNCH_CYCLES, EFR32_MVP_SPARSE_MIN_LAUNCHES
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import (
    baseline_pipeline,
    compile_source_to_execution_ir,
    licm_pipeline,
    optimized_pipeline,
)
from tests.test_reference import ChainSNN


def _module(dims, mode, pipeline, steps=100):
    net = ChainSNN(dims=dims, betas=[0.9] * len(dims), thresholds=[1.0] * len(dims))
    return compile_source_to_execution_ir(
        parse_snntorch_model(net), num_timesteps=steps, input_temporal_mode=mode, pipeline=pipeline()
    )


SHD_LIKE = [(700, 64), (64, 20)]
SSC_LIKE = [(700, 64), (64, 64), (64, 35)]
MNIST_LIKE = [(784, 64), (64, 32), (32, 10)]


def test_default_output_unchanged():
    mod = _module(SHD_LIKE, TemporalMode.VARYING, optimized_pipeline)
    assert generate_efr32_mvp_schedule(mod) == generate_efr32_mvp_schedule(mod, sparse_mvp=False)
    assert "iscas_sp_launch" not in generate_efr32_mvp_schedule(mod)


@pytest.mark.parametrize("dims", [SHD_LIKE, SSC_LIKE])
@pytest.mark.parametrize("pipeline", [baseline_pipeline, optimized_pipeline])
def test_temporal_input_layer_is_sparse_hidden_layers_stay_dense(dims, pipeline):
    mod = _module(dims, TemporalMode.VARYING, pipeline)
    plans = mvp_sparse_plan(mod)
    assert sorted(plans) == [0]
    p = plans[0]
    assert (p.rows, p.cols, p.source, p.absorbed_convert) == (64, 700, "step_input", "convert0")
    code = generate_efr32_mvp_schedule(mod, sparse_mvp=True)
    assert "convert0: convert -> stage0 ABSORBED" in code
    assert "range-merged sparse" in code
    # every non-sparse layer keeps the SDK dense kernel call
    assert code.count("sl_math_mvp_matrix_vector_mult_f16(&g_mat[") == len(dims) - 1 + 1  # + verify-build recompute


def test_static_input_hoisted_affine_is_never_sparse():
    mod = _module(MNIST_LIKE, TemporalMode.INVARIANT, licm_pipeline, steps=10)
    assert mvp_sparse_plan(mod) == {}
    mod = _module(MNIST_LIKE, TemporalMode.INVARIANT, optimized_pipeline, steps=10)
    assert generate_efr32_mvp_schedule(mod, sparse_mvp=True) == generate_efr32_mvp_schedule(mod)


def test_size_gate_and_odd_width_fall_back_to_dense():
    assert mvp_sparse_plan(_module([(64, 64), (64, 20)], TemporalMode.VARYING, optimized_pipeline)) == {}
    assert mvp_sparse_plan(_module([(701, 64), (64, 20)], TemporalMode.VARYING, optimized_pipeline)) == {}
    rows, pairs = 64, 700 // 2
    assert rows * pairs >= EFR32_MVP_SPARSE_MIN_LAUNCHES * EFR32_MVP_LAUNCH_CYCLES


def test_merge_gap_is_hardware_derived_not_workload_tuned():
    assert sparse_merge_gap_pairs(64) == 5          # 454 // (64 + 24), inside the measured flat optimum 4..8
    assert sparse_merge_gap_pairs(256) == 1         # tall matrices: a launch is cheap relative to a pair
    assert sparse_merge_gap_pairs(8) == 14


def test_runtime_fallback_and_counters_present():
    code = generate_efr32_mvp_schedule(_module(SSC_LIKE, TemporalMode.VARYING, optimized_pipeline), sparse_mvp=True)
    assert "g_sparse_fallback_count[0]++" in code
    assert re.search(r"nr = 1; rs\[0\] = 0; re\[0\] = 350; span = 350;", code)
    for name in ("launch", "pair", "range", "empty", "fallback"):
        assert f"int iscas_debug_sparse_{name}_count(int layer)" in code
    assert "#ifdef ISCAS_SPARSE_VERIFY" in code
    assert "case 0: return 1;" in code and "case 1: return 1;" not in code


def test_final_launch_reload_advances_accumulator_index():
    # the defect found during the isolated benchmark must stay fixed
    code = generate_efr32_mvp_schedule(_module(SHD_LIKE, TemporalMode.VARYING, optimized_pipeline), sparse_mvp=True)
    assert "SLI_MVP_LOAD(0, SLI_MVP_R3, ISCAS_SP_A_ACC, last ? SLI_MVP_INCRDIM_COL : 0)" in code


def test_sparse_requires_mvp_engine():
    mod = _module(SHD_LIKE, TemporalMode.VARYING, optimized_pipeline)
    with pytest.raises(ValueError):
        Efr32Backend().generate(mod, engine="cpu", sparse_mvp=True)
    files = Efr32Backend().generate(mod, engine="mvp", sparse_mvp=True).files
    assert "iscas_sp_launch" in files["snn_generated_mvp.c"]
