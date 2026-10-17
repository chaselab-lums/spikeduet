"""Test helpers: build generated C with the host compiler, drive the portable-C harness over its
binary stdin/stdout protocol, run generated EFR32 engines on the host, and compute the reference
readout (summed final-layer membrane, then argmax) in Python. Imported by the test modules."""

import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import List, Sequence

import torch

from spikeduet.ir.model import SNNModel
from spikeduet.reference.simulator import run_sequence

CC = shutil.which("gcc") or shutil.which("clang")


def compile_portable_c(files: dict, out_dir: Path, exe_name: str = "harness") -> Path:
    """Writes `files` (a GeneratedArtifact.files mapping, must include
    `main.c`) to `out_dir` and compiles them with the host C compiler
    using `-std=c99 -Wall -Wextra -Werror` (warnings are errors -- the
    generated code must be clean, not just "happens to compile"). Returns
    the path to the compiled executable.
    """
    assert CC is not None, "no host C compiler (gcc/clang) found on PATH"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(content)

    exe = out_dir / (exe_name + (".exe" if os.name == "nt" else ""))
    sources = [str(out_dir / "portable_runtime.c"), str(out_dir / "main.c")]
    cmd = [CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *sources, "-o", str(exe), "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"compile failed:\n{result.stdout}\n{result.stderr}"
    return exe


def run_harness(exe: Path, samples: Sequence[Sequence[float]]) -> List[int]:
    """Feeds `samples` (each a flat float sequence -- already flattened
    to [T][features] for temporal models by the caller) to the compiled
    harness binary in one batched process invocation and returns the
    predicted class for each sample, in order.
    """
    payload = struct.pack("i", len(samples))
    for sample in samples:
        payload += struct.pack(f"{len(sample)}f", *sample)

    result = subprocess.run([str(exe)], input=payload, capture_output=True)
    assert result.returncode == 0, f"harness run failed: {result.stderr}"

    n = len(samples)
    return list(struct.unpack(f"{n}i", result.stdout))


def strip_c_comments(text: str) -> str:
    """Strips C comments, so a symbol that only appears in a comment does not count as code."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def read_affine_exec_counts(files: dict, out_dir: Path, input_values: Sequence[float], num_layers: int) -> List[int]:
    """Compiles `files` (a PortableCBackend GeneratedArtifact.files
    mapping, WITHOUT the harness) alongside a small debug driver that
    calls `iscas_model_init()`/`iscas_model_run()` once on `input_values`
    and prints each layer's `iscas_model_debug_affine_exec_count(l)` --
    used to prove the generated preheader/loop schedule actually executes
    each layer's affine the expected number of times.
    """
    return read_exec_counts(files, out_dir, input_values, num_layers, "affine")


def read_exec_counts(
    files: dict, out_dir: Path, input_values: Sequence[float], num_layers: int, kind: str
) -> List[int]:
    """Generalizes `read_affine_exec_counts` to any of the debug counters
    `spikeduet/backend/portable_c.py` exposes: `kind` in
    `{"affine", "lif", "convert"}` -> `iscas_model_debug_{kind}_exec_count(l)`.
    `"convert"` is the staging-copy counter used to prove
    `SpikeStagingFusionPass` actually eliminates a fused edge's copy at
    runtime (0 executions), not just at the IR/text level -- see
    `tests/test_backend_fusion.py`.
    """
    assert kind in ("affine", "lif", "convert")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(content)

    driver = "\n".join(
        [
            "#include <stdio.h>",
            '#include "portable_runtime.h"',
            "int main(void) {",
            "    iscas_model_init();",
            f"    float x[{len(input_values)}] = {{{', '.join(f'{v}f' for v in input_values)}}};",
            "    iscas_model_run(x);",
            f"    for (int l = 0; l < {num_layers}; l++) {{",
            f'        printf("%d\\n", iscas_model_debug_{kind}_exec_count(l));',
            "    }",
            "    return 0;",
            "}",
        ]
    )
    (out_dir / "count_driver.c").write_text(driver)

    exe = out_dir / ("count_driver" + (".exe" if os.name == "nt" else ""))
    cmd = [
        CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        str(out_dir / "portable_runtime.c"), str(out_dir / "count_driver.c"),
        "-o", str(exe), "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"compile failed:\n{result.stdout}\n{result.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"count driver failed: {run.stderr}"
    return [int(line) for line in run.stdout.split() if line.strip()]


def python_reference_predict(ir: SNNModel, input_sequence: Sequence[torch.Tensor]) -> int:
    """The same readout policy `spikeduet/backend/portable_c.py` implements
    in C: accumulate the final LIF layer's membrane over every timestep in
    `input_sequence`, then argmax. This is the "compiler reference" this
    project's HIL/host-C fixtures have always compared physical firmware
    against -- never re-derived here,
    only reproduced in Python for comparison against the portable-C
    backend's own C implementation of the same policy.
    """
    _, _, traces = run_sequence(ir, input_sequence)
    mem_sum = torch.zeros_like(traces[0].mem[-1])
    for trace in traces:
        mem_sum = mem_sum + trace.mem[-1]
    return int(torch.argmax(mem_sum[0]).item())


SNN_H_PATH = Path(__file__).resolve().parents[1] / "runtime" / "efr32" / "snn.h"


def host_snn_run(files: dict, frames, workdir: Path, defines=(), stub: str = None) -> dict:
    """Host-compiles a generated EFR32 engine together with `runtime/efr32/snn.h` and streams
    `frames` ([N, T, C] float array) through it.

    Returns per-sample predictions, the summed final-layer membranes (the argmax input) and the
    per-layer mismatch counts reported by an `ISCAS_SPARSE_VERIFY` build. `stub` replaces
    `sl_math_mvp.h` for MVP engines.
    """
    import numpy as np

    assert CC is not None, "no host C compiler (gcc/clang) found on PATH"
    workdir.mkdir(parents=True, exist_ok=True)
    engine = files.get("snn_generated_cpu.c") or files.get("snn_generated_mvp.c") or files["snn.c"]
    (workdir / "model_data.h").write_text(files["model_data.h"])
    (workdir / "snn.c").write_text(engine)
    (workdir / "snn.h").write_text(SNN_H_PATH.read_text())
    if stub:
        (workdir / "sl_math_mvp.h").write_text(stub)
    frames = np.asarray(frames, dtype=np.float32)
    frames.tofile(workdir / "x.bin")
    _, steps, width = frames.shape
    if "iscas_debug_cpu_sparse_verify_mismatch" in engine:
        verify_decl = "int iscas_debug_cpu_sparse_verify_mismatch(int);"
    else:
        verify_decl = "static int iscas_debug_cpu_sparse_verify_mismatch(int l) { (void)l; return 0; }"
    (workdir / "main.c").write_text(f"""#include "snn.c"
#include <stdio.h>
{verify_decl}
static float x[{steps}][{width}];
int main(int argc, char **argv) {{
    FILE *f = fopen(argv[1], "rb");
    snn_init();
    while (fread(x, sizeof(x), 1, f) == 1) {{
        iscas_infer_reset(x[0]);
        for (int t = 0; t < {steps}; t++) iscas_infer_step(x[t]);
        printf("%d", iscas_infer_readout());
        for (int j = 0; j < ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features; j++) printf(" %.9g", (double)g_mem_sum[j]);
        for (int l = 0; l < ISCAS_NUM_LAYERS; l++) printf(" %d", iscas_debug_cpu_sparse_verify_mismatch(l));
        printf("\\n");
    }}
    return 0;
}}
""")
    exe = workdir / ("host.exe" if os.name == "nt" else "host")
    build = subprocess.run([CC, "-std=gnu11", "-O2", "-w", *[f"-D{d}" for d in defines], "-I", str(workdir),
                            str(workdir / "main.c"), "-o", str(exe), "-lm"], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-3000:]
    out = subprocess.run([str(exe), str(workdir / "x.bin")], capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in out.splitlines() if line.strip()]
    n_layers = int(re.search(r"#define ISCAS_NUM_LAYERS\s+(\d+)", files["model_data.h"]).group(1))
    n_out = len(rows[0]) - 1 - n_layers
    return {
        "pred": np.array([int(r[0]) for r in rows]),
        "memsum": np.array([[float(v) for v in r[1:1 + n_out]] for r in rows]),
        "verify": np.array([[int(v) for v in r[1 + n_out:]] for r in rows]),
    }

