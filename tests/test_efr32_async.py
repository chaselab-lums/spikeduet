"""Step-API pipelining of a range-sparse MVP hidden affine (spikeduet/backend/efr32/async_mvp.py),
the merge-gap override and SRAM-resident LIF parameters. The schedule, the chain logic and every buffer are the
generated code; only the MVP register helpers are replaced by the host twin of the range kernel."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.efr32.mvp_sparse import ASYNC_MVP_HW_HELPERS, HOST_RANGE_KERNEL_STUB, SPARSE_MVP_HELPERS
from tools.hil.firmware_build import MVP_HOST_STUB
from tests.test_efr32_cpu_sparse import _fold_module

CC = shutil.which("gcc") or shutil.which("clang")
SNN_H = Path("runtime/efr32/snn.h").read_text()
DIMS = [(24, 64), (64, 64), (64, 10)]
T = 12
HYB = dict(engine="mvp", sparse_mvp=True, force_sparse_mvp_layers=frozenset({1}), cpu_sparse_layers=frozenset({0, 2}),
           cpu_sparse_weight_placement={0: "sram", 2: "sram"}, mvp_flash_weight_layers=frozenset({1}), fold_output_affine=True)


def _gen(**kw):
    _, mod = _fold_module(DIMS, steps=T, thr=2.0)      # sparse hidden spikes: several ranges per timestep
    return Efr32Backend().generate(mod, **{**HYB, **kw}).files


def _frames(n=6, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, T, 24)) < 0.25).astype(np.float32)
    x[:, 3] = 0.0            # an all-zero input frame in every inference
    x[1, 5:] = 0.0           # an inference that ends in silence (empty MVP steps, including the last one)
    return x


def _host(files, tmp_path, frames, defines=(), n_in=24, n_out=10, layer=1):
    """Runs every inference back to back, then an ABORTED inference (half the steps, no readout) followed by a full
    one; returns the printed score bit patterns (one line per completed inference)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = files["snn_generated_mvp.c"]
    assert SPARSE_MVP_HELPERS in src
    src = src.replace(SPARSE_MVP_HELPERS, HOST_RANGE_KERNEL_STUB).replace(ASYNC_MVP_HW_HELPERS, "")
    (tmp_path / "model_data.h").write_text(files["model_data.h"])
    (tmp_path / "snn.c").write_text(src)
    (tmp_path / "snn.h").write_text(SNN_H)
    (tmp_path / "sl_math_mvp.h").write_text(MVP_HOST_STUB)
    for h in ("sl_mvp.h", "em_device.h"):
        (tmp_path / h).write_text("#include <stdint.h>\n#include <string.h>\n")
    n = len(frames)
    frames.astype(np.float32).tofile(tmp_path / "x.bin")
    (tmp_path / "main.c").write_text("\n".join([
        '#include "snn.c"', "#include <stdio.h>",
        f"static float x[{n}][{T}][{n_in}];",
        "static void show(int p) { printf(\"%d\", p); for (int j = 0; j < " + str(n_out) + "; j++) { unsigned u; memcpy(&u, &g_mem_sum[j], 4); printf(\" %08x\", u); }",
        "    printf(\" %d %d%c\", iscas_debug_sparse_launch_count(" + str(layer) + "), g_sparse_mvp_fault, 10); }",
        "int main(int argc, char **argv) {", "    FILE *f = fopen(argv[1], \"rb\"); if (fread(x, sizeof(x), 1, f) != 1) return 1; snn_init();",
        f"    for (int s = 0; s < {n}; s++) {{ iscas_infer_reset(x[s][0]); for (int t = 0; t < {T}; t++) iscas_infer_step(x[s][t]); show(iscas_infer_readout()); }}",
        f"    iscas_infer_reset(x[2][0]); for (int t = 0; t < {T // 2}; t++) iscas_infer_step(x[2][t]);   /* aborted: never read out */",
        f"    iscas_infer_reset(x[0][0]); for (int t = 0; t < {T}; t++) iscas_infer_step(x[0][t]); show(iscas_infer_readout());",
        "    return 0;", "}"]))
    exe = tmp_path / ("h" + (".exe" if os.name == "nt" else ""))
    r = subprocess.run([CC, "-std=gnu11", "-O2", "-w", *[f"-D{d}" for d in defines], "-I", str(tmp_path), str(tmp_path / "main.c"), "-o", str(exe), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = subprocess.run([str(exe), str(tmp_path / "x.bin")], capture_output=True, text=True, check=True).stdout.splitlines()
    return [l for l in rows if not l.startswith("W16FLASH")]     # boot check line of ISCAS_SPARSE_VERIFY builds


def test_async_is_opt_in_and_default_text_is_unchanged():
    base = _gen()["snn_generated_mvp.c"]
    for token in ("iscas_sp_start", "iscas_sp_poll", "g_async", "g_lifp_", "__DMB"):
        assert token not in base
    assert _gen(async_mvp=False, sparse_merge_gap=None, lif_params_sram=frozenset())["snn_generated_mvp.c"] == base


def test_async_structure():
    c = _gen(async_mvp=True)["snn_generated_mvp.c"]
    step = c[c.index("void iscas_infer_step"):c.index("int iscas_infer_readout")]
    ro = c[c.index("int iscas_infer_readout"):c.index("int run_snn_inference")]
    assert "iscas_sp_launch(g_weight_f16[1]" not in step                      # no blocking launch loop left for the async layer
    i_start, i_tail, i_wait = step.index("iscas_sp_begin_chain("), step.index("if (g_async_pending[1])"), step.index("iscas_sp_wait_chain(); iscas_sp_exit();")
    assert i_start < i_tail < i_wait                                         # start, deferred tail, wait
    assert step.count("iscas_sp_poll();") == 2 and "j0 += 32" in step        # output loop + LIF, polled every 32 neurons
    assert "lif1: DEFERRED (async_mvp)" in step
    assert ro.index("if (g_async_pending[1])") < ro.index("folded output affine (layer 2), once per inference")   # flush first
    assert "g_async_pending[1] = 0; g_async.active = 0;" in c[c.index("void iscas_infer_reset"):c.index("void iscas_infer_step")]
    assert "__DMB();" in c and "g_async_prev[1] = g_cur_f16[1]; g_cur_f16[1] = t_;" in step
    assert "if (f & ISCAS_SP_FAULTS) { g_sparse_mvp_fault = 1; }" in c


def test_async_legality():
    _, mod = _fold_module(DIMS, steps=T)
    with pytest.raises(ValueError, match="not a Cortex-M33 layer"):     # chain mode: a second MVP layer downstream
        Efr32Backend().generate(mod, **{**HYB, "fold_output_affine": False, "cpu_sparse_layers": frozenset({0}),
                                        "cpu_sparse_weight_placement": {0: "sram"}}, async_mvp=True)
    with pytest.raises(ValueError, match="exactly one range-sparse"):
        Efr32Backend().generate(mod, **{**HYB, "sparse_mvp": False, "force_sparse_mvp_layers": frozenset()}, async_mvp=True)
    with pytest.raises(ValueError, match="engine='mvp'"):
        Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, async_mvp=True)
    with pytest.raises(ValueError, match="not range-sparse"):
        Efr32Backend().generate(mod, **HYB, sparse_merge_gap={0: 2})


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_async_is_bitwise_equal_to_blocking(tmp_path):
    fr = _frames()
    gap0 = {"sparse_merge_gap": {1: 0}}                                       # no merging: several launches per timestep
    ref = _host(_gen(**gap0), tmp_path / "block", fr)
    assert len(ref) == len(fr) + 1 and ref[-1] == ref[0]                     # the aborted inference left nothing behind
    assert all(l.split()[-1] == "0" for l in ref)
    assert any(int(l.split()[-2]) > T for l in ref)                          # multi-launch chains are exercised
    a = _gen(async_mvp=True, **gap0)
    for defs in (["HOST_ASYNC_DONE_AFTER=1"], ["HOST_ASYNC_DONE_AFTER=3"], ["HOST_ASYNC_DONE_AFTER=1000000"], ["ISCAS_ASYNC_NO_OVERLAP=1"],
                 ["ISCAS_SPARSE_VERIFY=1"]):
        assert _host(a, tmp_path / ("a_" + defs[0].replace("=", "_")), fr, defs) == ref, defs


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_merge_gap_and_sram_parameters_do_not_change_the_result(tmp_path):
    fr = _frames(seed=3)
    ref = _host(_gen(), tmp_path / "block", fr)
    strip = lambda rows: [" ".join(l.split()[:-2]) for l in rows]             # scores only (the launch count changes with the gap)
    g = _gen(sparse_merge_gap={1: 0})
    assert "<= 0) { span +=" in g["snn_generated_mvp.c"]
    assert strip(_host(g, tmp_path / "gap0", fr)) == strip(ref)
    s = _gen(async_mvp=True, sparse_merge_gap={1: 2}, lif_params_sram=frozenset({0, 1}))
    c = s["snn_generated_mvp.c"]
    assert "iscas_param_at(g_lifp_beta[1], L->beta_len, j)" in c and "(g_lifp_bias[1] ? g_lifp_bias[1][j] : 0.0f)" in c
    assert strip(_host(s, tmp_path / "all", fr, ["HOST_ASYNC_DONE_AFTER=2"])) == strip(ref)


def test_lif_params_sram_on_the_cpu_engine():
    _, mod = _fold_module(DIMS, steps=T)
    base = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, fold_output_affine=True).files["snn_generated_cpu.c"]
    s = Efr32Backend().generate(mod, engine="cpu", sparse_cpu=True, fold_output_affine=True, lif_params_sram=frozenset({0, 1})).files["snn_generated_cpu.c"]
    assert "g_lifp_" not in base
    assert s.count("iscas_param_at(g_lifp_beta[") == 2 and "memcpy(g_lifp_beta[0], L->beta" in s


# ---- chain mode: the asynchronous affine is the input layer, the deferred tail is the rest of the timestep (SHD / SSC shape)
CHAIN_DIMS = [(240, 64), (64, 32), (32, 12)]
CHAIN = dict(engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}))


def _chain_gen(**kw):
    from tests.test_efr32_cpu_sparse import _module
    _, mod = _module(CHAIN_DIMS, steps=T, thr=0.6)
    return Efr32Backend().generate(mod, **{**CHAIN, **kw}).files


def test_chain_mode_structure():
    base = _chain_gen()["snn_generated_mvp.c"]
    assert "g_async" not in base
    c = _chain_gen(async_mvp=True)["snn_generated_mvp.c"]
    assert "@@ASYNC" not in c
    step = c[c.index("void iscas_infer_step"):c.index("int iscas_infer_readout")]
    ro = c[c.index("int iscas_infer_readout"):c.index("int run_snn_inference")]
    i_start, i_tail, i_wait = step.index("iscas_sp_begin_chain("), step.index("if (g_async_pending[0])"), step.index("iscas_sp_wait_chain(); iscas_sp_exit();")
    assert i_start < i_tail < i_wait
    tail = step[i_tail:i_wait]
    for token in ("lif0: lif layer 0", "affine1: affine layer 1", "lif1: lif layer 1", "affine2: affine layer 2", "lif2: lif layer 2",
                  "g_mem_sum[j] += g_mem[last][j];"):
        assert token in tail and token not in step[i_wait:], token           # the whole rest of the step is deferred
        assert token in ro                                                    # ... and flushed by the readout
    assert ro.index("g_mem_sum[j] += g_mem[last][j];") < ro.index("int predicted_class = 0;")


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_chain_mode_is_bitwise_equal_to_blocking(tmp_path):
    rng = np.random.default_rng(5)
    fr = (rng.random((5, T, 240)) < 0.03).astype(np.float32)
    fr[:, 2] = 0.0; fr[1, 6:] = 0.0                                           # empty frames, and an inference ending in silence
    for r in range(5):                                                         # clustered activity: several ranges per step
        fr[r, :, 40:52] = (rng.random((T, 12)) < 0.5)
    kw = dict(n_in=240, n_out=12, layer=0)
    ref = _host(_chain_gen(), tmp_path / "block", fr, **kw)
    assert len(ref) == len(fr) + 1 and ref[-1] == ref[0]
    assert any(int(l.split()[-2]) > T for l in ref)
    a = _chain_gen(async_mvp=True)
    for defs in (["HOST_ASYNC_DONE_AFTER=1"], ["HOST_ASYNC_DONE_AFTER=3"], ["HOST_ASYNC_DONE_AFTER=1000000"], ["ISCAS_ASYNC_NO_OVERLAP=1"]):
        assert _host(a, tmp_path / ("a_" + defs[0].replace("=", "_")), fr, defs, **kw) == ref, defs


def test_critical_path_options_are_opt_in():
    base = _chain_gen(async_mvp=True)["snn_generated_mvp.c"]
    assert "Register-only variant" not in base and "between row blocks" not in base and "j0 += 32" in base
    c = _chain_gen(async_mvp=True, async_poll_neurons=8, async_fast_wait=True, async_poll_m33=True)["snn_generated_mvp.c"]
    assert "Register-only variant" in c and "j0 += 8)" in c and "j0 += 32" not in c
    k1 = c[c.index("static void iscas_csp_acc1("):c.index("static void iscas_csp_acc2(")]
    k0 = c[c.index("static void iscas_sp_launch("):c.index("static void iscas_sp_start(")]
    assert k1.count("iscas_sp_poll();") == 1 and "iscas_sp_poll" not in k0      # 32 rows = 2 blocks -> 1 poll between them
    with pytest.raises(ValueError, match="need async_mvp=True"):
        _chain_gen(async_fast_wait=True)
    with pytest.raises(ValueError, match=">= 1"):
        _chain_gen(async_mvp=True, async_poll_neurons=0)


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_critical_path_options_are_bitwise_equal_to_blocking(tmp_path):
    rng = np.random.default_rng(7)
    fr = (rng.random((4, T, 240)) < 0.03).astype(np.float32)
    fr[:, 3] = 0.0
    for r in range(4):
        fr[r, :, 100:112] = (rng.random((T, 12)) < 0.5)
    kw = dict(n_in=240, n_out=12, layer=0)
    ref = _host(_chain_gen(), tmp_path / "block", fr, **kw)
    a = _chain_gen(async_mvp=True, async_poll_neurons=8, async_fast_wait=True, async_poll_m33=True)
    for defs in (["HOST_ASYNC_DONE_AFTER=1"], ["HOST_ASYNC_DONE_AFTER=5"], ["HOST_ASYNC_DONE_AFTER=1000000"], ["ISCAS_ASYNC_NO_OVERLAP=1"]):
        assert _host(a, tmp_path / ("c_" + defs[0].replace("=", "_")), fr, defs, **kw) == ref, defs
    fr2 = _frames(seed=9)                                                      # fold mode
    gap0 = {"sparse_merge_gap": {1: 0}}
    ref2 = _host(_gen(**gap0), tmp_path / "fblock", fr2)
    f = _gen(async_mvp=True, async_poll_neurons=8, async_fast_wait=True, **gap0)
    for defs in (["HOST_ASYNC_DONE_AFTER=1"], ["HOST_ASYNC_DONE_AFTER=1000000"]):
        assert _host(f, tmp_path / ("f_" + defs[0].replace("=", "_")), fr2, defs) == ref2, defs
