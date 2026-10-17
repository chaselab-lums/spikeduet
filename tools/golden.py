"""Fingerprints generated C across many backend configurations, ignoring comments and whitespace.

Used while refactoring: record a baseline, change the generators, then check that every
configuration still produces the same code.

    python -m tools.golden record golden.json
    python -m tools.golden check golden.json
"""
import hashlib
import json
import re
import sys

import torch

from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.portable_c import PortableCBackend
from spikeduet.baselines.mlp import (MLPWeights, generate_model_data_h, generate_snn_c_cpu, generate_snn_c_cpu_temporal,
                                  generate_snn_c_mvp, generate_snn_c_mvp_temporal)
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.pass_manager import PIPELINES, compile_source_to_execution_ir
from tests.test_reference import ChainSNN


_TOKEN = re.compile(r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""", re.S)


def normalize(code: str) -> str:
    """Drops comments and collapses whitespace; string and character literals are kept verbatim."""
    def keep(m):
        t = m.group(0)
        return " " if t.startswith("/") else t
    return " ".join(_TOKEN.sub(keep, code).split())


def _module(dims, steps, mode, pipeline, **chain):
    torch.manual_seed(0)
    n = len(dims)
    kw = dict(betas=[0.9] * n, thresholds=[1.0] * n, reset_delays=[False] * n)
    kw.update(chain)
    ir = parse_snntorch_model(ChainSNN(dims=dims, **kw))
    return compile_source_to_execution_ir(ir, num_timesteps=steps, input_temporal_mode=mode,
                                          pipeline=PIPELINES[pipeline]())


def configurations():
    temporal = TemporalMode.VARYING
    for pipe in PIPELINES:
        for mode in (TemporalMode.INVARIANT, temporal):
            m = _module([(16, 12), (12, 8), (8, 4)], 6, mode, pipe)
            yield f"portable/{pipe}/{mode.name}", PortableCBackend().generate(m, include_harness=True).files
            yield f"efr32/header/{pipe}/{mode.name}", Efr32Backend().generate(m).files
            yield f"efr32/cpu/{pipe}/{mode.name}", Efr32Backend().generate(m, engine="cpu").files
            yield f"efr32/mvp/{pipe}/{mode.name}", Efr32Backend().generate(m, engine="mvp").files
    wide = _module([(700, 64), (64, 20)], 10, temporal, "optimized")
    three = _module([(700, 64), (64, 64), (64, 35)], 10, temporal, "optimized",
                    reset_mechanisms=["subtract", "subtract", "none"])
    fold = _module([(30, 20), (20, 18), (18, 5)], 12, temporal, "optimized", betas=[0.9, 0.9, 0.0],
                   reset_mechanisms=["subtract", "subtract", "none"])
    gen = Efr32Backend().generate
    yield "mvp/sparse", gen(wide, engine="mvp", sparse_mvp=True).files
    yield "mvp/sparse/prof", gen(wide, engine="mvp", sparse_mvp=True, stage_profile=True).files
    yield "mvp/hybrid", gen(three, engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}),
                            stage_profile=True).files
    yield "mvp/static_io", gen(three, engine="mvp", mvp_static_io_layers=frozenset({1, 2})).files
    yield "mvp/window", gen(wide, engine="mvp", dense_window=5).files
    yield "mvp/lean", gen(wide, engine="mvp", dense_lean=True).files
    yield "mvp/full_width", gen(three, engine="mvp", mvp_full_width_layers=frozenset({1})).files
    yield "mvp/flash_w", gen(three, engine="mvp", mvp_flash_weight_layers=frozenset({0, 1, 2})).files
    yield "mvp/force_hidden", gen(three, engine="mvp", sparse_mvp=True, force_sparse_mvp_layers=frozenset({1})).files
    yield "mvp/async_chain", gen(three, engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}),
                                 async_mvp=True, stage_profile=True).files
    yield "mvp/async_chain_cp", gen(three, engine="mvp", sparse_mvp=True, cpu_sparse_layers=frozenset({1, 2}),
                                    async_mvp=True, async_fast_wait=True, async_poll_m33=True,
                                    async_poll_neurons=8).files
    yield "mvp/async_fold", gen(fold, engine="mvp", sparse_mvp=True, force_sparse_mvp_layers=frozenset({1}),
                                cpu_sparse_layers=frozenset({0, 2}), fold_output_affine=True, async_mvp=True,
                                sparse_merge_gap={1: 2}, stage_profile=True).files
    yield "cpu/sparse", gen(three, engine="cpu", sparse_cpu=True, stage_profile=True).files
    yield "cpu/placement", gen(three, engine="cpu", sparse_cpu=True,
                               cpu_sparse_weight_placement={0: "sram", 1: "flash", 2: "sram"}).files
    yield "cpu/pinned", gen(three, engine="cpu", sparse_cpu=True,
                            cpu_sparse_weight_placement={0: "sram", 1: "flash", 2: "sram"},
                            cpu_sparse_pinned_columns={1: [0, 3, 5, 9]}).files
    yield "cpu/fold", gen(fold, engine="cpu", sparse_cpu=True, fold_output_affine=True).files
    yield "cpu/fold_dense", gen(fold, engine="cpu", fold_output_affine=True).files
    yield "cpu/lif_sram", gen(three, engine="cpu", sparse_cpu=True, lif_params_sram=frozenset({0, 1})).files

    torch.manual_seed(1)
    dims = [24, 20, 18, 5]
    pairs = list(zip(dims, dims[1:]))
    mlp = MLPWeights(dims=dims, weights=[torch.randn(b, a).tolist() for a, b in pairs],
                     biases=[torch.randn(b).tolist() for _, b in pairs])
    yield "mlp/static", {"h": generate_model_data_h(mlp), "cpu": generate_snn_c_cpu(mlp),
                         "mvp": generate_snn_c_mvp(mlp)}
    yield "mlp/temporal", {"cpu": generate_snn_c_cpu_temporal(mlp, 8),
                           "cpu_blocks": generate_snn_c_cpu_temporal(mlp, 8, per_layer_blocks=True),
                           "mvp": generate_snn_c_mvp_temporal(mlp, 8)}
    yield "mlp/fair", {"a": generate_snn_c_mvp_temporal(mlp, 8, static_io_layers=(1, 2), event_input_layer=True,
                                                        reuse_empty_frames=True, stage_profile=True),
                       "b": generate_snn_c_mvp_temporal(mlp, 8, static_io_layers=(1, 2), event_input_layer=True,
                                                        reuse_empty_frames=True, fold_output_affine=True,
                                                        mvp_flash_weight_layers=frozenset({2})),
                       "c": generate_snn_c_mvp_temporal(mlp, 8, static_io_layers=(1,), event_input_layer=False,
                                                        reuse_empty_frames=True, mvp_split_rows={1: 10})}


def fingerprints():
    result = {}
    for name, files in configurations():
        for fname, text in sorted(files.items()):
            if fname.endswith((".c", ".h")) or fname in ("h", "cpu", "mvp", "a", "b", "c", "cpu_blocks"):
                result[f"{name}:{fname}"] = hashlib.sha256(normalize(text).encode()).hexdigest()[:16]
    return result


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    current = fingerprints()
    if cmd == "record":
        json.dump(current, open(path, "w"), indent=1, sort_keys=True)
        print(f"recorded {len(current)} fingerprints")
        return
    golden = json.load(open(path))
    bad = [k for k in golden if current.get(k) != golden[k]]
    missing = sorted(set(golden) ^ set(current))
    print(f"{len(golden) - len(bad)}/{len(golden)} identical")
    for k in bad + missing:
        print("DIFF", k)
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
