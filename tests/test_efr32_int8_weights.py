"""Opt-in int8 weight storage for the EFR32 dense-MVP engine (`Efr32Backend().generate(..., engine="mvp", int8_weights=True)`):
int8 flash tables + one float scale per output row, expanded to the FP16 MVP weight copy in `snn_init()`.
The MVP itself is unchanged (FP16 matvec); only the stored representation differs."""
import shutil
import subprocess

import numpy as np
import pytest
import torch

from spikeduet.backend.c_common import quantize_rows_int8
from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, optimized_pipeline
from tests.test_reference import ChainSNN

CC = shutil.which("gcc") or shutil.which("clang")
DIMS = [(16, 8), (8, 4)]


def _module(seed=0):
    torch.manual_seed(seed)
    net = ChainSNN(dims=DIMS, betas=[0.9] * len(DIMS), thresholds=[1.0] * len(DIMS))
    ir = parse_snntorch_model(net)
    return ir, compile_source_to_execution_ir(ir, num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT,
                                              pipeline=optimized_pipeline())


def test_default_output_unchanged():
    _, mod = _module()
    default = Efr32Backend().generate(mod, engine="mvp").files
    explicit = Efr32Backend().generate(mod, engine="mvp", int8_weights=False).files
    assert default == explicit
    assert "iscas_qweight_t" not in default["model_data.h"] and "ISCAS_QWEIGHTS" not in default["snn_generated_mvp.c"]


def test_int8_structure():
    _, mod = _module()
    files = Efr32Backend().generate(mod, engine="mvp", int8_weights=True).files
    header, src = files["model_data.h"], files["snn_generated_mvp.c"]
    for i in range(len(DIMS)):
        assert f"static const int8_t iscas_layer{i}_weight_q[" in header
        assert f"static const float iscas_layer{i}_weight_scale[" in header
        assert f"static const float iscas_layer{i}_weight[" not in header        # no FP32 weight array left in flash
    assert "ISCAS_QWEIGHTS" in header and "ISCAS_QWEIGHTS[l]" in src
    assert "L->weight" not in src                                                 # nothing reads the (NULL) FP32 pointer
    base = Efr32Backend().generate(mod, engine="mvp").files["snn_generated_mvp.c"]
    assert src.count("sl_math_mvp_matrix_vector_mult_f16(") == base.count("sl_math_mvp_matrix_vector_mult_f16(")


def test_row_quantizer():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(6, 9)).astype(np.float32) * np.array([[0.01], [1], [3], [0.2], [5], [1]], dtype=np.float32)
    w[3] = 0.0                                                                    # all-zero row
    q, s = quantize_rows_int8(w)
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert s[3] == 1.0 and not q[3].any()
    for r in (0, 1, 2, 4, 5):
        assert np.abs(q[r]).max() == 127
        assert np.abs(w[r] - q[r].astype(np.float32) * s[r]).max() <= s[r] * 0.5 * (1 + 1e-5)


@pytest.mark.parametrize("kwargs", [dict(engine="cpu"), dict(engine="mvp", sparse_mvp=True),
                                    dict(engine="mvp", cpu_sparse_layers=frozenset({1})),
                                    dict(engine="mvp", mvp_flash_weight_layers=frozenset({0})),
                                    dict(engine="mvp", dense_window=2)])
def test_int8_rejects_incompatible_options(kwargs):
    _, mod = _module()
    with pytest.raises(ValueError):
        Efr32Backend().generate(mod, int8_weights=True, **kwargs)


@pytest.mark.skipif(CC is None, reason="needs a host C compiler")
def test_host_compiled_dequantization_matches_numpy(tmp_path):
    ir, mod = _module(seed=3)
    header = Efr32Backend().generate(mod, engine="mvp", int8_weights=True).files["model_data.h"]
    (tmp_path / "model_data.h").write_text(header)
    (tmp_path / "t.c").write_text(
        '#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include "model_data.h"\n'
        "int main(void){ for (int l = 0; l < ISCAS_NUM_LAYERS; l++){ const iscas_layer_t *L = &ISCAS_LAYERS[l];\n"
        "  const iscas_qweight_t *Q = &ISCAS_QWEIGHTS[l];\n"
        "  for (int i = 0; i < L->out_features * L->in_features; i++){ float v = (float)Q->q[i] * Q->scale[i / L->in_features];\n"
        "    uint32_t b; memcpy(&b, &v, 4); printf(\"%08x\\n\", b); } } return 0; }\n")
    exe = tmp_path / "t.exe"
    subprocess.run([CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-o", str(exe), str(tmp_path / "t.c")], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    got = np.array([int(x, 16) for x in out], dtype=np.uint32).view(np.float32)
    want = []
    for layer in ir.layers:
        if hasattr(layer, "weight"):
            q, s = quantize_rows_int8(layer.weight)
            want.append((q.astype(np.float32) * s[:, None]).reshape(-1))
    want = np.concatenate(want)
    assert got.shape == want.shape and np.array_equal(got.view(np.uint32), want.view(np.uint32))
