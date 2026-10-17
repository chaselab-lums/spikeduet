"""FP16 equivalence of the direct-FP16 spike write on fused MVP edges.

On a fused edge the LIF casts its spike straight to FP16 instead of first copying it to an FP32
staging buffer. A lossless copy cannot change what a later cast computes; this module checks it
with a real compiled cast (GCC/Clang ``_Float16``, the same IEEE-754 binary16 format as CMSIS
``float16_t``) over spike values from the reference simulator: binary, graded (scalar and per
neuron), inhibition, state quantization and a combination of all of them.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.reference.simulator import run_sequence
from tests.test_reference import ChainSNN

CC = shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler (gcc/clang) on PATH")


def _both_paths_bit_identical(values: List[float]) -> bool:
    """Compiles a tiny C program that, for each `values[i]`, computes the
    OLD path (FP32 identity copy, then cast to `_Float16`) and the NEW
    path (direct cast to `_Float16`) and compares their raw uint16 bit
    patterns. Returns True iff every pair matches exactly.
    """
    assert CC is not None
    n = len(values)
    literals = ", ".join(f"{v!r}f" for v in values)
    source = "\n".join(
        [
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <string.h>",
            "int main(void) {",
            f"    float vals[{n}] = {{{literals}}};",
            "    int mismatches = 0;",
            f"    for (int i = 0; i < {n}; i++) {{",
            "        float copied = vals[i];               /* staged: FP32 identity copy, then cast */",
            "        _Float16 old_cast = (_Float16)copied;  /* cast after the copy */",
            "        _Float16 new_cast = (_Float16)vals[i]; /* direct cast, no copy */",
            "        uint16_t a, b;",
            "        memcpy(&a, &old_cast, sizeof(a));",
            "        memcpy(&b, &new_cast, sizeof(b));",
            "        if (a != b) { mismatches++; }",
            "    }",
            "    printf(\"%d\\n\", mismatches);",
            "    return 0;",
            "}",
        ]
    )

    tmp = Path(tempfile.mkdtemp())
    src_path = tmp / "f16_equiv.c"
    exe_path = tmp / "f16_equiv.exe"
    src_path.write_text(source)

    compile_result = subprocess.run(
        [CC, "-std=c99", "-Wall", "-Wextra", "-Werror", str(src_path), "-o", str(exe_path)],
        capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, f"compile failed:\n{compile_result.stdout}\n{compile_result.stderr}"

    run_result = subprocess.run([str(exe_path)], capture_output=True, text=True)
    assert run_result.returncode == 0, f"run failed: {run_result.stderr}"
    mismatches = int(run_result.stdout.strip())
    return mismatches == 0


def _final_spike_values(model, num_timesteps: int) -> List[float]:
    """Real emitted-spike float32 values from the Python reference
    simulator's LAST layer, across `num_timesteps`, for a static input --
    exactly the kind of value the EFR32 MVP backend's LIFUpdateOp would
    materialize (directly or via the old staged path)."""
    ir = parse_snntorch_model(model)
    torch.manual_seed(0)
    x = torch.randn(1, ir.layers[0].in_features)
    inputs = [x] * num_timesteps
    _, _, traces = run_sequence(ir, inputs)
    values: List[float] = []
    for trace in traces:
        values.extend(trace.spikes[-1][0].tolist())
    return values


def test_binary_spikes_fp16_equivalence():
    model = ChainSNN(dims=[(4, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_graded_scalar_fp16_equivalence():
    model = ChainSNN(
        dims=[(4, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0],
        graded_spikes_factors=[0.25, 1.5],
    )
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_graded_per_neuron_fp16_equivalence():
    model = ChainSNN(
        dims=[(4, 4), (4, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0],
        graded_spikes_factors=[[0.25, 0.5, 1.5, 2.5], 1.0],
    )
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_inhibition_fp16_equivalence():
    """Inhibited layers keep the staged path (see
    `mvp_emitters.mvp_direct_fused_layer_indices`); this proves the underlying cast claim still holds for
    inhibition's own emitted spike values regardless, since the SAME
    "identity copy is a no-op before a cast" argument applies to any
    float32 value, whichever path actually uses it."""
    model = ChainSNN(dims=[(4, 4), (4, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0], inhibitions=[True, False])
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_state_quant_fp16_equivalence():
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    model = ChainSNN(dims=[(4, 4), (4, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0], state_quants=[q, False])
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_full_l3_l4_l5_composition_fp16_equivalence():
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    model = ChainSNN(
        dims=[(4, 4), (4, 3)],
        betas=[[0.0, 0.4, 0.8, 1.0], 0.9],
        thresholds=[[0.3, 0.6, 0.9, 1.2], 1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
        reset_mechanisms=["zero", "subtract"],
        reset_delays=[True, False],
        inhibitions=[True, False],
        state_quants=[q, False],
    )
    values = _final_spike_values(model, num_timesteps=8)
    assert _both_paths_bit_identical(values)


def test_arbitrary_magnitude_edge_values_fp16_equivalence():
    """Beyond real fixtures: explicit edge-case magnitudes (subnormal-ish,
    large, exactly-representable, negative) that a real deployment could
    plausibly produce via graded_spikes_factor scaling."""
    values = [0.0, -0.0, 1.0, -1.0, 0.000030517578125, 65504.0, 12345.6789, -0.1]
    assert _both_paths_bit_identical(values)
