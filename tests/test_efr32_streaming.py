"""Host-compiled checks of the EFR32 CPU engine's streaming functions.

The CPU engine has no vendor dependency, so it is compiled with the host compiler against
``runtime/efr32/snn.h``. For a static and a time-varying fixture it checks that
``run_snn_inference()`` matches the reference simulator, and that driving ``iscas_infer_reset()``,
``iscas_infer_step()`` per timestep and ``iscas_infer_readout()`` gives the same prediction, which
is the path ``runtime/efr32/app.c`` uses for temporal models.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import (
    baseline_pipeline, compile_source_to_execution_ir, optimized_pipeline,
)
from spikeduet.reference.simulator import run_sequence
from tests.test_reference import ChainSNN

CC = shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler (gcc/clang) on PATH")

SNN_H = Path("runtime/efr32/snn.h").read_text()


def _compile_and_run(module, pipeline, tmp_path, input_flat, input_dim, num_timesteps):
    artifact = Efr32Backend().generate(module, engine="cpu")
    out_dir = tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_data.h").write_text(artifact.files["model_data.h"])
    (out_dir / "snn.c").write_text(artifact.files["snn_generated_cpu.c"])
    (out_dir / "snn.h").write_text(SNN_H)

    steps_vectors = [input_flat[t * input_dim:(t + 1) * input_dim] for t in range(num_timesteps)]

    def c_array(vals):
        return "{" + ", ".join(f"{v!r}f" for v in vals) + "}"

    driver = "\n".join(
        [
            '#include "model_data.h"',
            '#include "snn.h"',
            "#include <stdio.h>",
            "int main(void) {",
            "    snn_init();",
            f"    float whole[{len(input_flat)}] = {c_array(input_flat)};",
            "    int pred_whole = run_snn_inference(whole);",
            "",
            f"    float steps[{num_timesteps}][{input_dim}] = {{",
            *[f"        {c_array(v)}," for v in steps_vectors],
            "    };",
            "    iscas_infer_reset(steps[0]);",
            f"    for (int t = 0; t < {num_timesteps}; t++) {{",
            "        iscas_infer_step(steps[t]);",
            "    }",
            "    int pred_stream = iscas_infer_readout();",
            "",
            '    printf("%d %d\\n", pred_whole, pred_stream);',
            "    return 0;",
            "}",
        ]
    )
    (out_dir / "main.c").write_text(driver)

    exe = out_dir / ("harness" + (".exe" if os.name == "nt" else ""))
    cmd = [
        CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        str(out_dir / "snn.c"), str(out_dir / "main.c"), "-o", str(exe), "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"compile failed:\n{result.stdout}\n{result.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"run failed: {run.stderr}"
    pred_whole, pred_stream = (int(x) for x in run.stdout.split())
    return pred_whole, pred_stream


def _python_reference_predict(ir, input_sequence):
    _, _, traces = run_sequence(ir, input_sequence)
    mem_sum = torch.zeros_like(traces[0].mem[-1])
    for trace in traces:
        mem_sum = mem_sum + trace.mem[-1]
    return int(torch.argmax(mem_sum[0]).item())


@pytest.mark.parametrize("pipeline_fn", [baseline_pipeline, optimized_pipeline])
def test_streaming_matches_whole_tensor_static_input(tmp_path, pipeline_fn):
    """MNIST-shaped: static input, both pipelines (optimized hoists
    layer 0's affine into the preheader -- iscas_infer_step() then never
    reads step_input directly for that layer, proving the (void)step_input
    fallback path also compiles/works)."""
    model = ChainSNN(dims=[(8, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)
    num_timesteps = 5
    torch.manual_seed(0)
    x = torch.randn(1, 8)
    module = compile_source_to_execution_ir(
        ir, num_timesteps=num_timesteps, input_temporal_mode=TemporalMode.INVARIANT, pipeline=pipeline_fn(),
    )
    flat = (x[0].tolist()) * num_timesteps  # static: same vector replayed
    pred_whole, pred_stream = _compile_and_run(module, pipeline_fn, tmp_path, flat, 8, num_timesteps)
    assert pred_whole == pred_stream

    expected = _python_reference_predict(ir, [x] * num_timesteps)
    assert pred_whole == expected


@pytest.mark.parametrize("pipeline_fn", [baseline_pipeline, optimized_pipeline])
def test_streaming_matches_whole_tensor_temporal_input(tmp_path, pipeline_fn):
    """SHD-shaped: genuinely time-varying input (a different vector per
    timestep) -- the exact case that motivated the streaming split."""
    model = ChainSNN(dims=[(8, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    ir = parse_snntorch_model(model)
    num_timesteps = 6
    torch.manual_seed(1)
    inputs = [torch.randn(1, 8) for _ in range(num_timesteps)]
    module = compile_source_to_execution_ir(
        ir, num_timesteps=num_timesteps, input_temporal_mode=TemporalMode.VARYING, pipeline=pipeline_fn(),
    )
    flat = []
    for t in range(num_timesteps):
        flat.extend(inputs[t][0].tolist())
    pred_whole, pred_stream = _compile_and_run(module, pipeline_fn, tmp_path, flat, 8, num_timesteps)
    assert pred_whole == pred_stream

    expected = _python_reference_predict(ir, inputs)
    assert pred_whole == expected
