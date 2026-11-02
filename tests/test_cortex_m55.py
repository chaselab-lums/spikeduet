"""Cortex-M55 backend (N2): host bit-exactness of every realization, and an M55 cross-compile check that
the Helium kernels really are MVE code (no fused multiply-add, no vector spills)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from spikeduet.backend.cortex_m55 import M55_CFLAGS, REALIZATIONS, CortexM55Backend
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import baseline_pipeline, compile_source_to_execution_ir, optimized_pipeline
from spikeduet.reference.simulator import readout_scores, run_sequence
from tests.backend_c_harness import CC, run_memsum
from tests.test_reference import ChainSNN

REPO = Path(__file__).resolve().parents[1]
ARM_BIN = Path(os.environ.get("ISCAS_M55_GCC_BIN", "C:/ST/STM32CubeCLT_1.22.0/GNU-tools-for-STM32/bin"))
ARM_GCC = shutil.which("arm-none-eabi-gcc", path=str(ARM_BIN)) or shutil.which("arm-none-eabi-gcc")
ARM_OBJDUMP = shutil.which("arm-none-eabi-objdump", path=str(ARM_BIN)) or shutil.which("arm-none-eabi-objdump")
needs_cc = pytest.mark.skipif(CC is None, reason="no host C compiler")
needs_arm = pytest.mark.skipif(ARM_GCC is None or ARM_OBJDUMP is None, reason="no arm-none-eabi toolchain")


def _module(dims, T, mode, pipeline, seed=0, **kw):
    torch.manual_seed(seed)
    net = ChainSNN(dims=dims, betas=[0.9] * len(dims), thresholds=[0.5] * len(dims), **kw).eval()
    ir = parse_snntorch_model(net)
    return ir, compile_source_to_execution_ir(ir, num_timesteps=T, input_temporal_mode=mode, pipeline=pipeline)


def _memsums(module, realization, samples, tmp_path, tag):
    art = CortexM55Backend().generate(module, realization=realization)
    return run_memsum(art.files, tmp_path / tag, samples, sources=("m55_runtime.c",))


@needs_cc
@pytest.mark.parametrize("mode,pipeline", [(TemporalMode.VARYING, optimized_pipeline()),
                                           (TemporalMode.VARYING, baseline_pipeline()),
                                           (TemporalMode.INVARIANT, optimized_pipeline())])
def test_every_realization_is_bit_exact_to_scalar(tmp_path, mode, pipeline):
    dims = [(21, 37), (37, 18), (18, 5)]            # widths that are not multiples of 4 or 16
    T = 7
    ir, mod = _module(dims, T, mode, pipeline)
    n_in = 21 * (T if mode is TemporalMode.VARYING else 1)
    g = torch.Generator().manual_seed(1)
    samples = [((torch.rand(n_in, generator=g) < 0.3).float()).tolist() for _ in range(5)]          # binary
    samples += [((torch.rand(n_in, generator=g) * 2 - 0.5)).tolist() for _ in range(3)]             # real-valued
    ref = _memsums(mod, "scalar_dense", samples, tmp_path, "ref")
    for r in REALIZATIONS[1:]:
        assert _memsums(mod, r, samples, tmp_path, r) == ref, r
    mixed = {0: "helium_event", 1: "scalar_event", 2: "helium_dense"}
    assert _memsums(mod, mixed, samples, tmp_path, "mixed") == ref
    # and the scalar build agrees with the compiler reference (exact float literals, same argmax)
    import numpy as np
    for s, bits in zip(samples, ref):
        x = torch.tensor(s).reshape(-1, 21)
        xs = [x[t:t + 1] for t in range(T)] if mode is TemporalMode.VARYING else [x[0:1]] * T
        _, _, tr = run_sequence(ir, xs)
        want = readout_scores(ir, tr)[0].numpy()
        got = np.array(bits, dtype=np.uint32).view(np.float32)
        assert np.allclose(got, want, atol=1e-4) and got.argmax() == want.argmax()




def test_delays_only_scalar_and_unknown_realization_rejected():
    _, mod = _module([(6, 5), (5, 3)], 4, TemporalMode.VARYING, optimized_pipeline())
    with pytest.raises(ValueError, match="unknown realization"):
        CortexM55Backend().generate(mod, realization="npu")
    files = CortexM55Backend().generate(mod, realization="helium_event").files
    assert "ISCAS_M55_WSECT" in files["model_data.h"] and "ISCAS_M55_WT0" in files["m55_weights.h"]
    assert "g_m55_idx" in files["m55_runtime.c"]
    assert "g_m55_idx" not in CortexM55Backend().generate(mod, realization="helium_dense").files["m55_runtime.c"]


def _kernels(asm: str):
    out = {}
    for block in re.split(r"\n(?=[0-9a-f]+ <)", asm):
        m = re.match(r"[0-9a-f]+ <([^>]+)>:", block)
        if m:
            out[m.group(1)] = [l.split("\t", 1)[1] for l in block.splitlines()[1:] if "\t" in l]
    return out


def _cross(files, d: Path, *extra):
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_text(v)
    obj = d / "m55_runtime.o"
    res = subprocess.run([ARM_GCC, *M55_CFLAGS, *extra, "-Wall", "-Wextra", "-Werror", "-c", str(d / "m55_runtime.c"),
                          "-o", str(obj)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return _kernels(subprocess.run([ARM_OBJDUMP, "-d", "--no-show-raw-insn", str(obj)], capture_output=True, text=True).stdout)


@needs_arm
@pytest.mark.parametrize("workload", ["shd", "braille384"])
def test_m55_cross_compile_emits_mve_without_fma_or_spills(tmp_path, workload):
    dims = [(700, 64), (64, 20)] if workload == "shd" else [(16, 384), (384, 384), (384, 28)]
    _, mod = _module(dims, 7, TemporalMode.VARYING, optimized_pipeline())
    for r in ("helium_dense", "helium_event"):
        funcs = _cross(CortexM55Backend().generate(mod, realization=r).files, tmp_path / r)
        kern = [f for f in funcs if f.startswith(f"iscas_m55_{r}")]
        assert kern, sorted(funcs)
        body = funcs[kern[0]]
        ops = {i.split()[0] for i in body}
        assert {"vldrw.u32", "vadd.f32", "vmul.f32" if r == "helium_dense" else "vadd.f32"} <= ops, sorted(ops)
        assert any(i.startswith("vadd.f32") and " q" in i for i in body)
        assert not [i for f in funcs.values() for i in f if i.startswith(("vfma", "vfms"))], "fused multiply-add emitted"
        assert not [i for i in body if re.match(r"v(str|ldr)w\.\S+\s+q\d+, \[sp", i)], "vector register spilled"
    # the flag is load-bearing: with contraction allowed GCC fuses the scalar loop
    funcs = _cross(CortexM55Backend().generate(mod, realization="scalar_dense").files, tmp_path / "fast", "-ffp-contract=fast")
    assert any(i.startswith("vfma") for f in funcs.values() for i in f)
