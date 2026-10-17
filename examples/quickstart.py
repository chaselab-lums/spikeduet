"""End-to-end example: snnTorch model -> portable C -> host build -> compare with the reference.

    python examples/quickstart.py

Needs a host C compiler (gcc or clang) on PATH.
"""

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import snntorch as snn
import torch
import torch.nn as nn

from spikeduet import TemporalMode, compile_model, parse_snntorch_model, run_sequence

STEPS = 20


class TwoLayerSNN(nn.Module):
    """A ``[Linear -> Leaky]^2`` network written the way the frontend expects."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 12)
        self.lif1 = snn.Leaky(beta=0.9, reset_delay=False)
        self.fc2 = nn.Linear(12, 4)
        self.lif2 = snn.Leaky(beta=0.8, reset_delay=False)

    def forward(self, x):
        spk1, _ = self.lif1(self.fc1(x), self.lif1.mem)
        spk2, _ = self.lif2(self.fc2(spk1), self.lif2.mem)
        return spk2


def reference_prediction(ir, frames: torch.Tensor) -> int:
    """Readout used by every backend: final-layer membrane summed over all steps, then argmax."""
    _, _, traces = run_sequence(ir, [frames[t].unsqueeze(0) for t in range(STEPS)])
    return int(sum(trace.mem[-1] for trace in traces).argmax())


def main():
    torch.manual_seed(0)
    model = TwoLayerSNN().eval()
    with torch.no_grad():                      # stronger random weights so the network actually spikes
        for fc in (model.fc1, model.fc2):
            fc.weight.mul_(4.0)
    ir = parse_snntorch_model(model)

    artifact = compile_model(ir, backend="portable_c", num_timesteps=STEPS,
                             input_temporal_mode=TemporalMode.VARYING, pipeline="optimized",
                             include_harness=True)

    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise SystemExit("no host C compiler found; the generated files are in artifact.files")

    build = Path(tempfile.mkdtemp(prefix="spikeduet_"))
    artifact.write(build)
    exe = build / "model"
    subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(build),
                    str(build / "portable_runtime.c"), str(build / "main.c"), "-o", str(exe), "-lm"], check=True)

    samples = [(torch.rand(STEPS, 16) < 0.3).float() for _ in range(8)]
    payload = struct.pack("<i", len(samples)) + b"".join(
        struct.pack(f"<{STEPS * 16}f", *s.reshape(-1).tolist()) for s in samples)
    out = subprocess.run([str(exe)], input=payload, capture_output=True, check=True).stdout
    predicted_c = list(struct.unpack(f"<{len(samples)}i", out))
    predicted_py = [reference_prediction(ir, s) for s in samples]

    print("generated files:", ", ".join(sorted(artifact.files)), "->", build)
    print("C predictions:        ", predicted_c)
    print("reference predictions:", predicted_py)
    print("agreement:", sum(a == b for a, b in zip(predicted_c, predicted_py)), "/", len(samples))


if __name__ == "__main__":
    main()
