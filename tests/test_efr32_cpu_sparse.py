"""Reviewer campaign: event-driven Cortex-M33 affine realization, hybrid CPU/MVP layer
placement, hidden-spike-edge range-sparse MVP planning, and opt-in stage profiling
(spikeduet/backend/efr32/codegen.py).

The Cortex-M33 engine is plain C, so these tests host-compile it (gcc) and EXECUTE it:
the ISCAS_SPARSE_VERIFY build recomputes every sparse affine densely, exactly as the
scalar realization does, and compares bitwise on every timestep. MVP-side realizations
program MVP registers and are verified on silicon; here they are checked structurally.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import numpy as np
import torch

from spikeduet.backend.efr32.codegen import Efr32Backend, generate_efr32_cpu_schedule, generate_efr32_mvp_schedule
from spikeduet.backend.efr32.cost_model import mvp_dense_affine_cycles, mvp_sparse_affine_cycles, sparse_merge_gap_pairs
from spikeduet.backend.efr32.cpu_emitters import cpu_sparse_plan
from spikeduet.backend.efr32.mvp_emitters import mvp_sparse_plan
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import baseline_pipeline, compile_source_to_execution_ir, optimized_pipeline
from spikeduet.reference.simulator import run_sequence
from tests.test_reference import ChainSNN

CC = shutil.which("gcc") or shutil.which("clang")
SNN_H = Path("runtime/efr32/snn.h").read_text()

SHD_LIKE = [(700, 64), (64, 20)]
SSC_LIKE = [(700, 64), (64, 64), (64, 35)]
MNIST_LIKE = [(784, 64), (64, 32), (32, 10)]


def _module(dims, mode=TemporalMode.VARYING, pipeline=optimized_pipeline, steps=100, seed=0, thr=1.0):
    torch.manual_seed(seed)
    net = ChainSNN(dims=dims, betas=[0.9] * len(dims), thresholds=[thr] * len(dims))
    ir = parse_snntorch_model(net)
    return ir, compile_source_to_execution_ir(ir, num_timesteps=steps, input_temporal_mode=mode, pipeline=pipeline())


def test_default_outputs_unchanged():
    _, mod = _module(SHD_LIKE)
    assert generate_efr32_cpu_schedule(mod) == generate_efr32_cpu_schedule(mod, sparse_cpu=False, stage_profile=False)
    assert generate_efr32_mvp_schedule(mod) == generate_efr32_mvp_schedule(
        mod, sparse_mvp=False, cpu_sparse_layers=frozenset(), stage_profile=False)
    assert "iscas_csp_" not in generate_efr32_cpu_schedule(mod)
    assert "ISCAS_PROF_" not in generate_efr32_mvp_schedule(mod, sparse_mvp=True)


def test_cpu_sparse_plan_sources():
    _, mod = _module(SHD_LIKE)
    plans = cpu_sparse_plan(mod)
    assert sorted(plans) == [0, 1]
    assert (plans[0].source, plans[0].absorbed_convert) == ("step_input", "convert0")
    assert (plans[1].source, plans[1].absorbed_convert) == ("g_stage[1]", None)   # fused LIF0 output
    # static input: layer 0 is hoisted by LICM -> never a (per-timestep) sparse candidate
    _, mod = _module(MNIST_LIKE, TemporalMode.INVARIANT, steps=10)
    assert sorted(cpu_sparse_plan(mod)) == [1, 2]
    # unfused baseline pipeline: every hidden affine reads through a ConvertOp that gets absorbed
    _, mod = _module(SSC_LIKE, pipeline=baseline_pipeline)
    plans = cpu_sparse_plan(mod)
    assert [plans[i].absorbed_convert for i in (0, 1, 2)] == ["convert0", "convert1", "convert2"]
    assert plans[1].source == "g_spike[0]"


def test_cpu_kernel_is_register_blocked_and_flash_resident():
    _, mod = _module(SSC_LIKE)
    code = generate_efr32_cpu_schedule(mod, sparse_cpu=True)
    assert "static const float iscas_csp_wT0[44800]" in code             # 700 x 64, column-major, flash
    assert code.count("/* rows ") == 4 + 4 + 3                            # 64 -> 4x16, 64 -> 4x16, 35 -> 16+16+3
    assert "if (xv == 0)" in code and "a15 += w[15];" in code
    assert "convert0: convert -> stage0 ABSORBED" in code
    assert "malloc" not in code.split("void iscas_infer_reset")[1]        # no allocation during inference


def _run_host(files, tmp_path, frames, defines=()):
    """Compiles the generated CPU engine + a driver that streams `frames` ([T][N] lists); returns
    (prediction, per-layer verify mismatches, per-layer event counts)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model_data.h").write_text(files["model_data.h"])
    (tmp_path / "snn.c").write_text(files["snn_generated_cpu.c"])
    (tmp_path / "snn.h").write_text(SNN_H)
    rows = ",\n".join("{" + ", ".join(f"{v!r}f" for v in fr) + "}" for fr in frames)
    sparse = "iscas_debug_cpu_sparse_layer" in files["snn_generated_cpu.c"]
    driver = [
        '#include "model_data.h"', '#include "snn.h"', "#include <stdio.h>",
        "int iscas_debug_cpu_sparse_verify_mismatch(int); int iscas_debug_cpu_sparse_event_count(int);" if sparse else "",
        f"static float steps[{len(frames)}][{len(frames[0])}] = {{{rows}}};",
        "int main(void) {", "    snn_init();", "    iscas_infer_reset(steps[0]);",
        f"    for (int t = 0; t < {len(frames)}; t++) iscas_infer_step(steps[t]);",
        '    printf("%d", iscas_infer_readout());',
    ]
    if sparse:
        driver.append('    for (int l = 0; l < ISCAS_NUM_LAYERS; l++) printf(" %d %d", '
                      "iscas_debug_cpu_sparse_verify_mismatch(l), iscas_debug_cpu_sparse_event_count(l));")
    driver += ['    printf("\\n");', "    return 0;", "}"]
    (tmp_path / "main.c").write_text("\n".join(driver))
    exe = tmp_path / ("h" + (".exe" if os.name == "nt" else ""))
    cmd = [CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *[f"-D{d}" for d in defines],
           str(tmp_path / "snn.c"), str(tmp_path / "main.c"), "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    vals = list(map(int, out))
    return vals[0], vals[1::2], vals[2::2]


def _reference_predict(ir, frames):
    _, _, tr = run_sequence(ir, [torch.tensor([f]) for f in frames])
    return int(sum(t.mem[-1] for t in tr)[0].argmax())


@pytest.mark.skipif(CC is None, reason="no host C compiler")
@pytest.mark.parametrize("binary", [True, False])
def test_cpu_sparse_is_bitwise_equal_to_dense(tmp_path, binary):
    """Binary events exercise the add-only kernel; real-valued inputs the general kernel. Both must be
    bitwise equal to the dense scalar recompute on EVERY timestep and layer (verify build), and the
    prediction must equal the dense CPU engine's and the Python reference's."""
    dims = [(46, 20), (20, 18), (18, 5)]                      # odd block remainders (20 = 16+4, 18 = 16+2)
    ir, mod = _module(dims, steps=12, seed=3, thr=0.3)
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(12, 46, generator=g) < 0.15).float() if binary else torch.randn(12, 46, generator=g) * (torch.rand(12, 46, generator=g) < 0.2)
    frames = x.tolist()
    dense = Efr32Backend().generate(mod, engine="cpu").files
    sparse = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True).files
    pd, _, _ = _run_host(dense, tmp_path / "d", frames)
    ps, mism, events = _run_host(sparse, tmp_path / "s", frames, defines=["ISCAS_SPARSE_VERIFY"])
    assert mism == [0, 0, 0]
    assert events[0] == int((x.view(torch.int32) != 0).sum())   # by bit pattern: -0.0 counts as active
    assert ps == pd == _reference_predict(ir, frames)


def test_hybrid_places_hidden_layers_on_cpu():
    _, mod = _module(SSC_LIKE)
    code = generate_efr32_mvp_schedule(mod, sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}))
    assert "iscas_sp_launch" in code                                      # Linear0 stays range-sparse MVP
    assert "iscas_csp_acc1(" in code and "iscas_csp_acc2(" in code
    assert "g_in_f16[1][j] = (float16_t)spike" not in code                # LIF0 now writes the FP32 stage buffer
    assert "g_stage[1][j] = spike;" in code
    assert "if (l == 1 || l == 2) {" in code                              # no FP16 weight copy for CPU layers
    assert code.count("sl_math_mvp_matrix_vector_mult_f16(&g_mat[") == 1  # only the verify-build recompute of L0


def test_hidden_spike_edges_use_the_same_planner():
    _, mod = _module(SSC_LIKE)
    assert sorted(mvp_sparse_plan(mod)) == [0]                            # static policy unchanged
    forced = mvp_sparse_plan(mod, force_layers=frozenset({1, 2}))
    assert forced[1].fp16_source and forced[1].source == "g_in_f16[1]"
    assert forced[1].gap_pairs == sparse_merge_gap_pairs(64, fp16_source=True)
    # profile-guided: measured hidden activity keeps both hidden edges dense
    act = {0: (5.65, 62.0), 1: (1.57, 12.2), 2: (1.26, 11.2)}
    assert sorted(mvp_sparse_plan(mod, activity=act)) == [0]
    # a sparse enough hidden activity would be selected by the same rule
    assert 1 in mvp_sparse_plan(mod, activity={1: (1.0, 2.0)})
    code = generate_efr32_mvp_schedule(mod, sparse_mvp=True, force_sparse_mvp_layers=frozenset({1}))
    assert "FP16 source in place" in code and "if (xb[p] != 0u)" in code


def test_cost_model_reproduces_measured_linear0_fit():
    for L, P in ((0, 0), (4, 65), (10, 200)):
        assert abs(mvp_sparse_affine_cycles(64, 700, L, P, fp32_source=True) - (5462 + 454 * L + 87.9 * P)) < 25
    assert abs(mvp_dense_affine_cycles(64, 700, fp32_source=True, copy=True) - 34768) < 2


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_stage_profile_markers_compile_to_nothing(tmp_path):
    ir, mod = _module([(30, 12), (12, 4)], steps=6, seed=1, thr=0.4)
    frames = (torch.rand(6, 30, generator=torch.Generator().manual_seed(2)) < 0.3).float().tolist()
    plain = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True).files
    prof = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, stage_profile=True).files
    assert "ISCAS_PROF_LAP(" in prof["snn_generated_cpu.c"]
    for name in ("affine0.plan", "affine0.compute", "lif0", "affine1.plan", "lif1", "memsum", "readout", "reset"):
        assert f"stage: {name} */" in prof["snn_generated_cpu.c"]
    assert _run_host(plain, tmp_path / "a", frames)[0] == _run_host(prof, tmp_path / "b", frames)[0]
    mvp = generate_efr32_mvp_schedule(mod, sparse_mvp=True, stage_profile=True)
    assert "stage: affine1.mvp */" in mvp and "stage: lif0 */" in mvp


def test_dense_lean_and_window_realizations():
    from spikeduet.backend.efr32.mvp_emitters import dense_input_plan
    _, mod = _module(SHD_LIKE)
    plan = dense_input_plan(mod, 10)
    assert (plan.layer, plan.rows, plan.cols, plan.absorbed_convert, plan.window) == (0, 64, 700, "convert0", 10)
    lean = generate_efr32_mvp_schedule(mod, dense_lean=True)
    assert "g_in_f16[0][i] = (float16_t)step_input[i]" in lean and "convert0: convert -> stage0 ABSORBED" in lean
    assert "g_stage[0][i] = source[i]" not in lean                         # the FP32 identity copy is gone
    win = generate_efr32_mvp_schedule(mod, dense_window=10)
    assert "iscas_wn_matvec_batch(g_weight_f16[0], g_wn_x, g_wn_z, 64, 350, 10);" in win
    assert "if (++g_wn_n < 10)" in win and "for (int s = 0; s < 10; s++)" in win
    # the batched program is the SDK matvec's own per-row sequence: CLEAR; MACR2A; ADDR+STORE
    assert "SLI_MVP_OP(MACR2A)" in win and "SLI_MVP_OP(ADDR)" in win and "MACC" not in win
    with pytest.raises(ValueError):
        dense_input_plan(mod, 7)                                         # T=100 is not a multiple of 7
    with pytest.raises(ValueError):
        generate_efr32_mvp_schedule(mod, sparse_mvp=True, dense_lean=True)  # exclusive with sparse Linear0


def test_mlp_sparse_input_layer_shares_the_kernel():
    import ast
    import spikeduet.backend.efr32.mvp_sparse as kern
    from spikeduet.baselines.mlp import MLPWeights, generate_snn_c_mvp_temporal
    # neither the shared kernel nor the ANN generator may depend on the SNN IR / SNN backend
    for mod_path in ("spikeduet/backend/efr32/mvp_sparse.py", "spikeduet/backend/efr32/cpu_sparse.py",
                     "spikeduet/baselines/mlp.py"):
        tree = ast.parse(Path(mod_path).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
                [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(x and (x.startswith("spikeduet.ir") or x.startswith("spikeduet.frontend")
                              or x == "spikeduet.backend.efr32.codegen") for x in names), (mod_path, names)
    g = torch.Generator().manual_seed(0)
    dims = [700, 64, 64, 35]
    mlp = MLPWeights(dims=dims, weights=[torch.randn(dims[i + 1], dims[i], generator=g).tolist() for i in range(3)],
                     biases=[torch.randn(dims[i + 1], generator=g).tolist() for i in range(3)])
    dense = generate_snn_c_mvp_temporal(mlp, 100)
    sparse = generate_snn_c_mvp_temporal(mlp, 100, sparse_input_layer=True)
    assert dense == generate_snn_c_mvp_temporal(mlp, 100, sparse_input_layer=False)
    assert "iscas_sp_launch" not in dense and kern.SPARSE_MVP_HELPERS in sparse
    assert "p - (int)re[nr - 1] <= 5" in sparse                             # same FP32 merge-gap policy as the SNN
    assert "for (int l = 1; l < ISCAS_NUM_LAYERS; l++)" in sparse           # hidden layers unchanged (dense MVP)
    assert "g_act[0][j] = (acc < 0.0f) ? 0.0f : acc;" in sparse             # same bias + ReLU after the kernel


def test_stage_profile_indices_are_unique_and_all_emitted():
    """Every registered stage has exactly one counter and appears in the emitted code (the loop body
    is rendered more than once internally; a stage must never be registered twice)."""
    import re
    cases = [generate_efr32_cpu_schedule(_module(SSC_LIKE)[1], sparse_cpu=True, stage_profile=True),
             generate_efr32_mvp_schedule(_module(SSC_LIKE)[1], sparse_mvp=True, cpu_sparse_layers=frozenset({2}), stage_profile=True),
             generate_efr32_mvp_schedule(_module(SSC_LIKE)[1], dense_window=10, stage_profile=True),
             generate_efr32_mvp_schedule(_module(MNIST_LIKE, TemporalMode.INVARIANT, steps=10)[1], stage_profile=True)]
    for code in cases:
        n, names = re.search(r"g_prof_names\[(\d+)\] = \{(.*?)\};", code).groups()
        names = re.findall(r'"([^"]+)"', names)
        assert len(names) == len(set(names)) == int(n)
        assert sorted({int(x) for x in re.findall(r"ISCAS_PROF_LAP\((\d+)\)", code)}) == list(range(int(n)))


def test_full_width_raw_launch_realization():
    """Dense execution through one full-width launch of the raw range kernel: no scan, no SDK call,
    same launch helpers; exclusive with any other realization of the same layer."""
    from spikeduet.backend.efr32.mvp_emitters import mvp_full_width_plans
    _, mod = _module(SSC_LIKE)
    fw = mvp_full_width_plans(mod, frozenset({0, 1, 2}))
    assert (fw[0].source, fw[0].absorbed_convert, fw[0].fp16_source) == ("step_input", "convert0", False)
    assert fw[1].fp16_source and fw[2].fp16_source
    code = generate_efr32_mvp_schedule(mod, mvp_full_width_layers=frozenset({0, 1, 2}))
    assert code.count("one full-width raw launch") == 3 and "iscas_sp_launch(g_weight_f16[1], g_in_f16[1], 0, 32, 1, 1);" in code
    import re
    outside_verify = re.sub(r"#ifdef ISCAS_SPARSE_VERIFY.*?#endif", "", code, flags=re.S)
    assert "sl_math_mvp_matrix_vector_mult_f16(&g_mat" not in outside_verify                               # SDK only in verify builds
    assert "if ((xb[2 * p] | xb[2 * p + 1]) != 0u)" not in code                                      # no scan
    with pytest.raises(ValueError):
        generate_efr32_mvp_schedule(mod, sparse_mvp=True, mvp_full_width_layers=frozenset({0}))


def test_static_io_placement_option_both_generators():
    """Placement experiment: the listed layers' FP16 MVP operand vectors are re-pointed to static .bss
    arrays after all heap allocations (so no other buffer moves); default output unchanged."""
    from spikeduet.baselines.mlp import MLPWeights
    _, mod = _module(SSC_LIKE)
    assert generate_efr32_mvp_schedule(mod) == generate_efr32_mvp_schedule(mod, mvp_static_io_layers=frozenset())
    code = generate_efr32_mvp_schedule(mod, sparse_mvp=True, mvp_static_io_layers=frozenset({1, 2}))
    assert "static float16_t g_io_in1[64]" in code and "static float16_t g_io_cur2[35]" in code
    init = code.split("void snn_init(void)")[1].split("void iscas_infer_reset")[0]
    assert init.index("g_in_f16[l] = (float16_t *)snn_alloc_check") < init.index("g_in_f16[1] = g_io_in1;")
    g = torch.Generator().manual_seed(1)
    dims = [700, 64, 64, 35]
    mlp = MLPWeights(dims=dims, weights=[torch.randn(dims[i + 1], dims[i], generator=g).tolist() for i in range(3)],
                     biases=[torch.randn(dims[i + 1], generator=g).tolist() for i in range(3)])
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    assert generate_snn_c_mvp_temporal(mlp, 100) == generate_snn_c_mvp_temporal(mlp, 100, static_io_layers=())
    for sp in (False, True):
        c = generate_snn_c_mvp_temporal(mlp, 100, sparse_input_layer=sp, static_io_layers=(1, 2))
        assert "g_in_f16[2] = g_io_in2; g_cur_f16[2] = g_io_cur2;" in c.split("void iscas_infer_reset")[0]


# --- temporal MLP baseline (spikeduet/baselines/mlp.py) ---

from tools.hil.firmware_build import MVP_HOST_STUB as _MVP_HOST_STUB   # host FP16 matvec stand-in (logic tests only)


def _random_mlp(dims, seed):
    from spikeduet.baselines.mlp import MLPWeights
    g = torch.Generator().manual_seed(seed)
    return MLPWeights(dims=dims, weights=[(torch.randn(dims[i + 1], dims[i], generator=g) * 0.4).tolist() for i in range(len(dims) - 1)],
                      biases=[(torch.randn(dims[i + 1], generator=g) * 0.1).tolist() for i in range(len(dims) - 1)])


def _run_mlp_host(mlp, code, tmp_path, frames, defines=()):
    """Compiles a generated temporal MLP engine with the host MVP stand-in; returns (prediction, reused frames,
    layer-0 events, layer-0 empty steps, layer-0 verify mismatches)."""
    from spikeduet.baselines.mlp import generate_model_data_h, generate_snn_h
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model_data.h").write_text(generate_model_data_h(mlp, temporal_steps=len(frames)))
    (tmp_path / "snn.h").write_text(generate_snn_h())
    (tmp_path / "sl_math_mvp.h").write_text(_MVP_HOST_STUB)
    (tmp_path / "snn.c").write_text(code)
    rows = ",\n".join("{" + ", ".join(f"{v!r}f" for v in fr) + "}" for fr in frames)
    (tmp_path / "main.c").write_text("\n".join([
        '#include "model_data.h"', '#include "snn.h"', "#include <stdio.h>",
        "int iscas_debug_mlp_reused_frames(void); int iscas_debug_cpu_sparse_event_count(int);",
        "int iscas_debug_cpu_sparse_empty_count(int); int iscas_debug_cpu_sparse_verify_mismatch(int);",
        f"static float steps[{len(frames)}][{len(frames[0])}] = {{{rows}}};",
        "int main(void) {", "    snn_init();", "    iscas_infer_reset(steps[0]);",
        f"    for (int t = 0; t < {len(frames)}; t++) iscas_infer_step(steps[t]);",
        '    printf("%d %d %d %d %d\\n", iscas_infer_readout(), iscas_debug_mlp_reused_frames(),',
        "           iscas_debug_cpu_sparse_event_count(0), iscas_debug_cpu_sparse_empty_count(0), iscas_debug_cpu_sparse_verify_mismatch(0));",
        "    return 0;", "}"]))
    extra = ""                                               # builds without these counters: driver placeholders
    if "iscas_debug_mlp_reused_frames" not in code:
        extra += "int iscas_debug_mlp_reused_frames(void) { return -1; }\n"
    if "iscas_debug_cpu_sparse_event_count" not in code:
        extra += "".join(f"int iscas_debug_cpu_sparse_{c}(int l) {{ (void)l; return -1; }}\n"
                         for c in ("event_count", "empty_count", "verify_mismatch"))
    (tmp_path / "snn.c").write_text(code + extra)
    exe = tmp_path / ("h" + (".exe" if os.name == "nt" else ""))
    cmd = [CC, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(tmp_path), *[f"-D{d}" for d in defines],
           str(tmp_path / "snn.c"), str(tmp_path / "main.c"), "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return list(map(int, subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()))


def test_fair_mlp_generator_defaults_and_structure():
    """Default output unchanged; the fair variant shares the SNN's M33 kernel, gives layer 0 no FP16 copy,
    fixes hidden-layer placement BEFORE the zero-frame cache is computed, and emits unique stage markers."""
    import re
    from spikeduet.backend.efr32.cpu_sparse import CPU_SPARSE_HELPERS
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    mlp = _random_mlp([24, 20, 18, 5], 0)
    base = generate_snn_c_mvp_temporal(mlp, 12)
    assert base == generate_snn_c_mvp_temporal(mlp, 12, event_input_layer=False, reuse_empty_frames=False, stage_profile=False)
    assert "iscas_csp_" not in base and "g_empty_logits" not in base
    fair = generate_snn_c_mvp_temporal(mlp, 12, static_io_layers=(1, 2), event_input_layer=True, reuse_empty_frames=True, stage_profile=True)
    assert CPU_SPARSE_HELPERS.strip("\n") in fair and "iscas_csp_acc0(g_csp_wT[0]" in fair
    init = fair.split("void snn_init(void)")[1].split("void iscas_infer_reset")[0]
    assert "if (l < 1) { continue; }" in init
    assert init.index("g_in_f16[2] = g_io_in2;") < init.index("memcpy(g_empty_logits, mlp_eval(g_zero_frame, 0)")
    n, names = re.search(r"g_prof_names\[(\d+)\] = \{(.*?)\};", fair).groups()
    names = re.findall(r'"([^"]+)"', names)
    assert len(names) == len(set(names)) == int(n) and {"affine0.plan", "reuse", "affine1.out", "logits"} <= set(names)
    with pytest.raises(AssertionError):
        generate_snn_c_mvp_temporal(mlp, 12, sparse_input_layer=True, event_input_layer=True)


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_fair_mlp_reuse_is_exact_and_event_layer_bitwise(tmp_path):
    """Empty-frame reuse must change no prediction (verify build recomputes every reused frame and compares the
    logits bitwise); the event-driven layer 0 must be bitwise equal to the dense FP32 recompute; the reuse and
    empty counters must equal the number of all-zero frames."""
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    dims = [24, 20, 18, 5]                                         # 20 = 16+4, 18 = 16+2 row blocks
    g = torch.Generator().manual_seed(5)
    for seed in range(3):
        mlp = _random_mlp(dims, seed)
        x = (torch.rand(12, 24, generator=g) < 0.2).float()
        x[torch.rand(12, generator=g) < 0.4] = 0.0                 # ~40% empty frames
        frames, n_empty = x.tolist(), int((x.sum(1) == 0).sum())
        kw = dict(static_io_layers=(1, 2), event_input_layer=True)
        reuse = generate_snn_c_mvp_temporal(mlp, 12, reuse_empty_frames=True, **kw)
        noreuse = generate_snn_c_mvp_temporal(mlp, 12, reuse_empty_frames=False, **kw)
        pr, reused, ev, empty, mism = _run_mlp_host(mlp, reuse, tmp_path / f"r{seed}", frames, ["ISCAS_SPARSE_VERIFY"])
        pn, _, ev_n, empty_n, mism_n = _run_mlp_host(mlp, noreuse, tmp_path / f"n{seed}", frames, ["ISCAS_SPARSE_VERIFY"])
        assert pr == pn and mism == mism_n == 0
        assert reused == empty == empty_n == n_empty and ev == ev_n == int(x.sum())
        # dense-MVP layer 0 with reuse: same logic, zero test by bit pattern
        dense_reuse = generate_snn_c_mvp_temporal(mlp, 12, reuse_empty_frames=True, static_io_layers=(1, 2))
        dense_noreuse = generate_snn_c_mvp_temporal(mlp, 12, reuse_empty_frames=False, stage_profile=True, static_io_layers=(1, 2))
        pdr, reused_d, _, _, mism_d = _run_mlp_host(mlp, dense_reuse, tmp_path / f"dr{seed}", frames, ["ISCAS_SPARSE_VERIFY"])
        pdn = _run_mlp_host(mlp, dense_noreuse, tmp_path / f"dn{seed}", frames)[0]
        assert pdr == pdn and reused_d == n_empty and mism_d == 0


# --- per-layer weight placement ---
def test_per_layer_weight_placement_codegen():
    """cpu_sparse_weight_placement fixes each event-driven layer's transposed-W residency in the generated code;
    None keeps the global ISCAS_CPU_SPARSE_WT_SRAM text byte-identical; the dict must name exactly the M33 layers."""
    _, mod = _module([(46, 20), (20, 18), (18, 5)], steps=4)
    base = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True).files["snn_generated_cpu.c"]
    assert base == Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=None).files["snn_generated_cpu.c"]
    mixed = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True,
                                    cpu_sparse_weight_placement={0: "sram", 1: "flash", 2: "sram"}).files["snn_generated_cpu.c"]
    init = mixed.split("void snn_init(void)")[1].split("\n}\n")[0]
    assert "ISCAS_CPU_SPARSE_WT_SRAM" not in init
    assert "g_csp_wT[1] = iscas_csp_wT1;" in init and "memcpy(wt, iscas_csp_wT0" in init and "memcpy(wt, iscas_csp_wT2" in init
    assert "memcpy(wt, iscas_csp_wT1" not in init
    hybrid = Efr32Backend().generate(mod, engine="mvp", cpu_sparse_layers=frozenset({1, 2}),
                                     cpu_sparse_weight_placement={1: "flash", 2: "sram"}).files["snn_generated_mvp.c"]
    assert "g_csp_wT[1] = iscas_csp_wT1;" in hybrid and "memcpy(wt, iscas_csp_wT2" in hybrid
    for bad in ({0: "sram", 1: "flash"}, {0: "sram", 1: "flash", 2: "sram", 3: "sram"}, {0: "sram", 1: "rom", 2: "sram"}):
        with pytest.raises(ValueError):
            Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=bad)
    with pytest.raises(ValueError):
        Efr32Backend().generate(mod, engine=None, cpu_sparse_weight_placement={0: "sram"})


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_mixed_weight_placement_is_bitwise_equal(tmp_path):
    """Residency never changes arithmetic: a mixed SRAM/flash build is bitwise equal to the dense recompute on every
    timestep (verify build) and predicts exactly what the all-SRAM (global define) build predicts."""
    dims = [(46, 20), (20, 18), (18, 5)]
    ir, mod = _module(dims, steps=12, seed=4, thr=0.3)
    g = torch.Generator().manual_seed(11)
    frames = (torch.rand(12, 46, generator=g) < 0.15).float().tolist()
    allsram = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True).files
    mixed = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True,
                                    cpu_sparse_weight_placement={0: "flash", 1: "sram", 2: "flash"}).files
    pa, ma, ea = _run_host(allsram, tmp_path / "a", frames, defines=["ISCAS_SPARSE_VERIFY", "ISCAS_CPU_SPARSE_WT_SRAM"])
    pm, mm, em = _run_host(mixed, tmp_path / "m", frames, defines=["ISCAS_SPARSE_VERIFY"])
    assert ma == mm == [0, 0, 0] and ea == em
    assert pa == pm == _reference_predict(ir, frames)


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_fair_mlp_flash_weight_layers(tmp_path):
    """MLP per-layer weight placement: FP16 tables in flash read in place by the MVP give the same predictions as the
    snn_init() SRAM copies; the event layer's residency is per layer too; defaults stay byte-identical."""
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    dims = [24, 20, 18, 5]
    kw = dict(static_io_layers=(1, 2), event_input_layer=True, reuse_empty_frames=True)
    mlp = _random_mlp(dims, 0)
    assert generate_snn_c_mvp_temporal(mlp, 12, **kw) == generate_snn_c_mvp_temporal(mlp, 12, mvp_flash_weight_layers=(), **kw)
    code = generate_snn_c_mvp_temporal(mlp, 12, mvp_flash_weight_layers=(2,), event_weights="flash", **kw)
    assert "static const uint16_t mlp_w16_l2[90]" in code and "mlp_w16_l1" not in code
    assert "g_csp_wT[0] = iscas_csp_wT0;" in code and "W16FLASH" in code
    with pytest.raises(AssertionError):
        generate_snn_c_mvp_temporal(mlp, 12, mvp_flash_weight_layers=(0,), **kw)      # layer 0 is the M33 event layer
    g = torch.Generator().manual_seed(9)
    for seed in range(3):
        mlp = _random_mlp(dims, seed)
        x = (torch.rand(12, 24, generator=g) < 0.2).float()
        frames = x.tolist()
        ref = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, **kw), tmp_path / f"s{seed}", frames)
        fl = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, mvp_flash_weight_layers=(1, 2), event_weights="flash", **kw),
                           tmp_path / f"f{seed}", frames)
        assert fl == ref


# --- hot-column pinning ---
def test_select_hot_columns():
    from spikeduet.backend.efr32.cpu_sparse import select_hot_columns
    assert select_hot_columns([5, 1, 9, 9, 0, 3], 3) == [0, 2, 3]          # ties -> lower index; result ascending
    assert select_hot_columns([1, 2], 5) == [0, 1] and select_hot_columns([1, 2], 0) == []


def test_pinned_columns_codegen_and_validation():
    _, mod = _module([(46, 20), (20, 18), (18, 5)], steps=4)
    pl = {0: "sram", 1: "flash", 2: "sram"}
    plain = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=pl).files["snn_generated_cpu.c"]
    assert plain == Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=pl,
                                            cpu_sparse_pinned_columns=None).files["snn_generated_cpu.c"]
    pin = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=pl,
                                  cpu_sparse_pinned_columns={1: [3, 0, 7]}).files["snn_generated_cpu.c"]
    assert "static const uint16_t iscas_csp_pin1[3] = {0, 3, 7};" in pin and "static const float *g_csp_colp1[20];" in pin
    assert "iscas_csp_accp1(g_csp_colp1, g_csp_idx[1]" in pin and "iscas_csp_acc0(g_csp_wT[0]" in pin
    assert "iscas_csp_accp0" not in pin and "const float *w = colp[ai[t]] + 16u;" in pin
    for bad_pl, bad_pin in ((pl, {0: [1]}),                       # layer 0 is SRAM-resident: nothing to pin
                            (None, {1: [1]}),                     # residency must be explicit
                            (pl, {1: [20]}), (pl, {1: []})):      # out of range / empty
        with pytest.raises(ValueError):
            Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=bad_pl,
                                    cpu_sparse_pinned_columns=bad_pin)


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_pinned_columns_are_bitwise_equal(tmp_path):
    """Pinning only moves where a column is read from: every timestep stays bitwise equal to the dense recompute and the
    prediction equals the unpinned build's, with hits and misses in the same step."""
    dims = [(46, 20), (20, 18), (18, 5)]
    ir, mod = _module(dims, steps=12, seed=6, thr=0.3)
    g = torch.Generator().manual_seed(13)
    frames = (torch.rand(12, 46, generator=g) < 0.2).float().tolist()
    pl = {0: "flash", 1: "flash", 2: "sram"}
    base = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=pl).files
    pinned = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, cpu_sparse_weight_placement=pl,
                                     cpu_sparse_pinned_columns={0: list(range(0, 46, 3)), 1: [0, 5, 6, 19]}).files
    pb, mb, eb = _run_host(base, tmp_path / "b", frames, defines=["ISCAS_SPARSE_VERIFY"])
    pp_, mp, ep = _run_host(pinned, tmp_path / "p", frames, defines=["ISCAS_SPARSE_VERIFY"])
    assert mb == mp == [0, 0, 0] and eb == ep
    assert pb == pp_ == _reference_predict(ir, frames)


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_fair_mlp_split_rows(tmp_path):
    """MLP partial residency (rows 0..R-1 SRAM, rows R.. flash, two launches) predicts exactly what the unsplit build does."""
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    dims = [24, 20, 18, 5]
    kw = dict(static_io_layers=(1, 2), event_input_layer=True, reuse_empty_frames=True)
    mlp = _random_mlp(dims, 1)
    code = generate_snn_c_mvp_temporal(mlp, 12, mvp_split_rows={1: 8}, **kw)
    assert "g_mat_tail[1], g_in_f16[1], g_cur_f16[1] + 8" in code and "mlp_w16_tail_l1[200]" in code
    for bad in ({1: 7}, {1: 0}, {1: 18}, {0: 4}):
        with pytest.raises(AssertionError):
            generate_snn_c_mvp_temporal(mlp, 12, mvp_split_rows=bad, **kw)
    g = torch.Generator().manual_seed(21)
    for seed in range(3):
        mlp = _random_mlp(dims, seed)
        frames = (torch.rand(12, 24, generator=g) < 0.2).float().tolist()
        ref = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, **kw), tmp_path / f"u{seed}", frames)
        sp = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, mvp_split_rows={1: 10, 2: 2}, **kw), tmp_path / f"s{seed}", frames)
        assert sp == ref


# --- output-affine folding ---
def _fold_module(dims, out_beta=0.0, out_reset="none", inhibit_hidden=False, steps=12, seed=0, thr=0.4):
    torch.manual_seed(seed)
    n = len(dims)
    net = ChainSNN(dims=dims, betas=[0.9] * (n - 1) + [out_beta], thresholds=[thr] * n,
                   reset_mechanisms=["subtract"] * (n - 1) + [out_reset],
                   inhibitions=[False] * (n - 2) + [inhibit_hidden, False])
    for lin in net.linears:
        torch.nn.init.normal_(lin.bias, 0.0, 0.3)             # nonzero biases everywhere
    ir = parse_snntorch_model(net)
    return ir, compile_source_to_execution_ir(ir, num_timesteps=steps, input_temporal_mode=TemporalMode.VARYING,
                                              pipeline=optimized_pipeline())


def test_fold_output_affine_legality():
    from spikeduet.backend.efr32.fold import output_fold_plan
    _, ok = _fold_module([(30, 20), (20, 18), (18, 5)])
    plan = output_fold_plan(ok)
    assert plan.layer == 2 and plan.rows == 5 and plan.cols == 18 and plan.producer_lif_id is not None and plan.steps == 12
    for kw, msg in ((dict(out_beta=0.9), "beta"), (dict(out_reset="subtract"), "reset"), (dict(out_reset="zero"), "reset")):
        _, bad = _fold_module([(30, 20), (20, 18), (18, 5)], **kw)
        with pytest.raises(ValueError, match=msg):
            output_fold_plan(bad)
    _, inh = _fold_module([(30, 20), (20, 18), (18, 5)], inhibit_hidden=True)
    assert output_fold_plan(inh).producer_lif_id is None              # inhibited producer -> separate accumulate loop
    with pytest.raises(ValueError):
        Efr32Backend().generate(ok, engine="mvp", fold_output_affine=True)
    base = Efr32Backend().generate(ok, engine="cpu", sparse_cpu=True).files["snn_generated_cpu.c"]
    assert base == Efr32Backend().generate(ok, engine="cpu", sparse_cpu=True, fold_output_affine=False).files["snn_generated_cpu.c"]


@pytest.mark.skipif(CC is None, reason="no host C compiler")
@pytest.mark.parametrize("sparse,inhibit", [(True, False), (False, False), (True, True)])
def test_fold_output_affine_matches_per_step(tmp_path, sparse, inhibit):
    """Folded vs per-step output affine on host-compiled firmware: identical predictions on every sample, summed
    logits equal to FP32 reassociation tolerance, folded kernel bitwise equal to the dense recompute (verify build);
    inputs include all-empty frames and nonzero biases."""
    from tests.backend_c_harness import host_snn_run as _host_snn_run
    _, mod = _fold_module([(30, 20), (20, 18), (18, 5)], inhibit_hidden=inhibit)
    g = torch.Generator().manual_seed(3)
    x = (torch.rand(40, 12, 30, generator=g) < 0.25).float()
    x[:, ::4] = 0.0                                                     # empty frames
    x[0] = 0.0                                                          # a fully empty sequence
    kw = dict(engine="cpu", sparse_cpu=sparse)
    a = _host_snn_run(Efr32Backend().generate(mod, **kw).files, x.numpy(), tmp_path / "a")
    b = _host_snn_run(Efr32Backend().generate(mod, fold_output_affine=True, **kw).files, x.numpy(), tmp_path / "b",
                      defines=["ISCAS_SPARSE_VERIFY"] if sparse else ())
    assert (a["pred"] == b["pred"]).all()
    assert np.allclose(a["memsum"], b["memsum"], rtol=1e-5, atol=1e-4)
    assert int(b["verify"].sum()) == 0


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_fair_mlp_fold_output_affine(tmp_path):
    """MLP folding: default text unchanged; folded readout = per-frame readout (host FP16 stand-in) on sequences with
    empty frames, including a fully empty one and the event-driven-layer-0 variant."""
    from spikeduet.baselines.mlp import generate_snn_c_mvp_temporal
    dims = [24, 20, 18, 5]
    g = torch.Generator().manual_seed(31)
    for event in (False, True):
        kw = dict(static_io_layers=(1, 2) if event else (0, 1, 2), event_input_layer=event, reuse_empty_frames=True)
        mlp = _random_mlp(dims, 2)
        assert generate_snn_c_mvp_temporal(mlp, 12, **kw) == generate_snn_c_mvp_temporal(mlp, 12, fold_output_affine=False, **kw)
        for seed in range(3):
            x = (torch.rand(12, 24, generator=g) < 0.2).float()
            x[::3] = 0.0
            if seed == 0:
                x[:] = 0.0
            frames = x.tolist()
            ref = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, **kw), tmp_path / f"o{event}{seed}", frames)
            fold = _run_mlp_host(mlp, generate_snn_c_mvp_temporal(mlp, 12, fold_output_affine=True, **kw), tmp_path / f"f{event}{seed}", frames)
            assert fold[0] == ref[0] and fold[1] == ref[1]          # prediction, reused-frame count


# --- Final-evaluation baseline: flash-resident FP16 MVP weights in the SNN backend (Efr32Backend mvp_flash_weight_layers) ---

def _run_mvp_snn_host(files, tmp_path, frames):
    """Host-compiles a generated dense-MVP SNN engine with the host MVP stand-in; returns the summed scores as exact
    hex-float strings (argmax input)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model_data.h").write_text(files["model_data.h"]); (tmp_path / "snn.c").write_text(files["snn_generated_mvp.c"])
    (tmp_path / "snn.h").write_text(SNN_H); (tmp_path / "sl_math_mvp.h").write_text(_MVP_HOST_STUB)
    rows = ",\n".join("{" + ", ".join(f"{v!r}f" for v in fr) + "}" for fr in frames)
    (tmp_path / "main.c").write_text("\n".join([
        '#include "snn.c"', "#include <stdio.h>", f"static float steps[{len(frames)}][{len(frames[0])}] = {{{rows}}};",
        "int main(void) {", "    snn_init();", "    iscas_infer_reset(steps[0]);",
        f"    for (int t = 0; t < {len(frames)}; t++) iscas_infer_step(steps[t]);", "    iscas_infer_readout();",
        "    for (int j = 0; j < ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features; j++) printf(\"%a \", (double)g_mem_sum[j]);",
        "    return 0;", "}"]))
    exe = tmp_path / ("h" + (".exe" if os.name == "nt" else ""))
    r = subprocess.run([CC, "-std=gnu11", "-O2", "-w", "-I", str(tmp_path), str(tmp_path / "main.c"), "-o", str(exe), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()


def test_mvp_flash_weight_layers_structure():
    _, mod = _module([(46, 20), (20, 18), (18, 5)], steps=4)
    base = Efr32Backend().generate(mod, engine="mvp", mvp_static_io_layers=frozenset({1, 2})).files["snn_generated_mvp.c"]
    assert base == Efr32Backend().generate(mod, engine="mvp", mvp_static_io_layers=frozenset({1, 2}),
                                           mvp_flash_weight_layers=frozenset()).files["snn_generated_mvp.c"]
    assert "iscas_w16_flash" not in base
    fl = Efr32Backend().generate(mod, engine="mvp", mvp_static_io_layers=frozenset({1, 2}),
                                 mvp_flash_weight_layers=frozenset({0, 2})).files["snn_generated_mvp.c"]
    assert "static const uint16_t iscas_w16_l0[920]" in fl and "static const uint16_t iscas_w16_l2[90]" in fl
    assert "iscas_w16_l1[" not in fl and "{iscas_w16_l0, 0, iscas_w16_l2}" in fl.replace("iscas_w16_flash[ISCAS_NUM_LAYERS] = ", "")
    with pytest.raises(ValueError):
        Efr32Backend().generate(mod, engine="cpu", mvp_flash_weight_layers=frozenset({0}))
    with pytest.raises(ValueError):      # an M33 layer has no FP16 MVP weights (a flash table next to M33 layers is legal)
        Efr32Backend().generate(mod, engine="mvp", mvp_flash_weight_layers=frozenset({0}), cpu_sparse_layers=frozenset({0}))


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_mvp_flash_weight_layers_bitwise_equal_to_sram_copy(tmp_path):
    """The flash table is the numpy FP16 rounding of the FP32 weights == the (float16_t) cast snn_init() would make,
    so the summed scores are bitwise identical."""
    rng = np.random.default_rng(3)
    _, mod = _module([(46, 20), (20, 18), (18, 5)], steps=6)
    frames = [list(map(float, (rng.random(46) < 0.3).astype(np.float32))) for _ in range(6)]
    a = _run_mvp_snn_host(Efr32Backend().generate(mod, engine="mvp", mvp_static_io_layers=frozenset({1, 2})).files, tmp_path / "sram", frames)
    b = _run_mvp_snn_host(Efr32Backend().generate(mod, engine="mvp", mvp_static_io_layers=frozenset({1, 2}),
                                                  mvp_flash_weight_layers=frozenset({0, 1, 2})).files, tmp_path / "flash", frames)
    assert a == b and len(a) == 5
