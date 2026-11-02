"""Core delay, adaptive-threshold, weighted-readout and EFR32 speed tests; no workload-specific frontend."""

import importlib.util

import os

import shutil

import subprocess

import tempfile

from pathlib import Path

import numpy as np

import pytest

import torch

from spikeduet.backend.api import compile_model

from spikeduet.backend.efr32.codegen import Efr32Backend

from spikeduet.backend.schedule_common import extension_usage

from spikeduet.ir.execution import DelayPushOp, LIFUpdateOp, TemporalMode

from spikeduet.ir.execution_verifier import ExecutionIRVerificationError, verify_execution_ir

from spikeduet.ir.model import AdaptiveThresholdSpec, LIFLayer, LinearLayer, SNNModel

from spikeduet.ir.pass_manager import baseline_pipeline, compile_source_to_execution_ir, optimized_pipeline

from spikeduet.ir.verifier import IRVerificationError, verify_module_ir

from spikeduet.reference.simulator import _lif_step, initialize_state, readout_scores, run_sequence, step

from tests.backend_c_harness import CC, compile_portable_c, run_harness

REPO = Path(__file__).resolve().parents[1]

needs_cc = pytest.mark.skipif(CC is None, reason="no host C compiler")

STEPS = 9

def tiny_ir(seed=0, *, adaptation=True, delays=True, weighted=True, steps=STEPS) -> SNNModel:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    layer0 = LinearLayer("l0", 3, 5, r(5, 3) * 1.2, r(5) * 0.1)
    lif0 = LIFLayer("a", beta=[0.9, 0.5, 0.7, 0.3, 0.8], threshold=[-0.3, 0.2, 0.5, 0.15, 0.4],
                    reset_mechanism="subtract", reset_delay=True, spike_state=True)
    d = torch.randint(0, 4, (4, 5), generator=g) if delays else None
    layer1 = LinearLayer("l1", 5, 4, r(4, 5) * 1.5, r(4) * 0.1, synapse_delays=d)
    lif1 = LIFLayer("b", beta=[0.8, 0.6, 0.9, 0.5], threshold=[0.3, -0.2, 0.4, 0.25], reset_mechanism="subtract",
                    reset_delay=True, spike_state=True,
                    adaptation=AdaptiveThresholdSpec(rho=(0.9, 0.8, 0.95, 0.7), a=(0.4, 0.2, 0.6, 0.3)) if adaptation else None)
    if not adaptation:
        lif1.spike_state = True
    layer2 = LinearLayer("l2", 4, 3, r(3, 4), r(3) * 0.1)
    lif2 = LIFLayer("c", beta=[0.9, 0.95, 0.85], threshold=[1.0], reset_mechanism="none")
    return SNNModel([layer0, lif0, layer1, lif1, layer2, lif2], num_steps=steps,
                    readout_weights=tuple(r(steps).tolist()) if weighted else None)

def tiny_inputs(n=64, steps=STEPS, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, steps, 3, generator=g) * 1.5

def ref_scores(ir, x):
    _, _, tr = run_sequence(ir, [x[:, t] for t in range(x.shape[1])])
    return readout_scores(ir, tr)

def test_tiny_model_verifies():
    verify_module_ir(tiny_ir())

@pytest.mark.parametrize("mutate,needle", [
    (lambda ir: setattr(ir.layers[3], "reset_delay", False), "adaptation"),
    (lambda ir: setattr(ir.layers[3], "reset_mechanism", "zero"), "reset_mechanism"),
    (lambda ir: setattr(ir.layers[3], "inhibition", True), "inhibition"),
    (lambda ir: setattr(ir.layers[3], "graded_spikes_factor", [2.0]), "graded_spikes_factor"),
    (lambda ir: setattr(ir.layers[3], "spike_state", False), "spike_state is False"),
    (lambda ir: setattr(ir.layers[2], "synapse_delays", ir.layers[2].synapse_delays.float()), "integer"),
    (lambda ir: setattr(ir.layers[2], "synapse_delays", ir.layers[2].synapse_delays[:, :3]), "shape"),
    (lambda ir: setattr(ir.layers[2], "synapse_delays", ir.layers[2].synapse_delays - 5), "negative"),
    (lambda ir: setattr(ir.layers[2], "synapse_delays", ir.layers[2].synapse_delays + 300), "255"),
    (lambda ir: setattr(ir, "readout_weights", (0.1, 0.2)), "readout_weights"),
    (lambda ir: setattr(ir.layers[3], "adaptation", AdaptiveThresholdSpec(rho=(1.5,), a=(0.1,))), "rho"),
    (lambda ir: setattr(ir.layers[3], "adaptation", AdaptiveThresholdSpec(rho=(0.5,), a=(-0.1,))), "a must be"),
])
def test_verifier_rejects_invalid_extension_use(mutate, needle):
    ir = tiny_ir()
    mutate(ir)
    with pytest.raises(IRVerificationError) as exc:
        verify_module_ir(ir)
    assert needle in str(exc.value)

def test_spike_state_with_zero_reset_rejected():
    ir = tiny_ir()
    ir.layers[1].reset_mechanism = "zero"
    with pytest.raises(IRVerificationError):
        verify_module_ir(ir)

def test_spike_state_differs_from_membrane_derived_reset_only_through_negative_threshold():
    """The stored previous spike (0 at t=0) and the stock `mem > threshold` indicator agree from t=1 on; they differ at
    t=0 exactly when a threshold is negative (a resting membrane is then 'above threshold')."""
    cur = torch.tensor([[0.4, 0.4]])
    for theta, expect_equal in ((0.5, True), (-0.5, False)):
        lif = LIFLayer("n", beta=[0.9], threshold=[theta], reset_mechanism="subtract", reset_delay=True)
        stock_spk, stock_mem = _lif_step(lif, cur[:, :1].expand(1, 1), torch.zeros(1, 1))
        lif_s = LIFLayer("n", beta=[0.9], threshold=[theta], reset_mechanism="subtract", reset_delay=True, spike_state=True)
        ir = SNNModel([LinearLayer("l", 1, 1, torch.ones(1, 1)), lif_s])
        _, _, tr = run_sequence(ir, [cur[:, :1]])
        assert (torch.allclose(tr[0].mem[0], stock_mem)) == expect_equal

def test_delays_gather_the_right_history():
    """Synapse (j, i) with delay d sees input i from d steps ago; identity weights expose the gather directly."""
    delays = torch.tensor([[0, 1], [2, 0]])
    lin = LinearLayer("l", 2, 2, torch.ones(2, 2), None, synapse_delays=delays)
    ir = SNNModel([lin, LIFLayer("n", beta=[0.0], threshold=[99.0], reset_mechanism="none")])
    xs = [torch.tensor([[float(t + 1), float(10 * (t + 1))]]) for t in range(4)]
    _, _, tr = run_sequence(ir, xs)
    for t in range(4):
        x = lambda k, i: xs[k][0, i].item() if k >= 0 else 0.0
        want0 = x(t, 0) + x(t - 1, 1)          # row 0: delays (0, 1)
        want1 = x(t - 2, 0) + x(t, 1)          # row 1: delays (2, 0)
        assert tr[t].currents[0][0].tolist() == pytest.approx([want0, want1])

def test_alif_adaptation_changes_the_output_and_is_nonvacuous():
    x = tiny_inputs(16) * 2.0
    with_alif, without = tiny_ir(adaptation=True), tiny_ir(adaptation=False)
    s1, s2 = ref_scores(with_alif, x), ref_scores(without, x)
    assert not torch.allclose(s1, s2)
    _, _, tr = run_sequence(with_alif, [x[:, t] for t in range(STEPS)])
    assert sum(float(t.spikes[1].sum()) for t in tr) > 0, "hidden layer never fires: the test would be vacuous"

def test_weighted_readout_is_the_weighted_membrane_sum():
    ir = tiny_ir()
    x = tiny_inputs(8)
    _, _, tr = run_sequence(ir, [x[:, t] for t in range(STEPS)])
    want = sum(w * t.mem[-1] for w, t in zip(ir.readout_weights, tr))
    assert torch.allclose(readout_scores(ir, tr), want, atol=1e-6)
    ir.readout_weights = None
    assert torch.allclose(readout_scores(ir, tr), sum(t.mem[-1] for t in tr), atol=1e-6)

def test_ordinary_models_keep_their_original_state_layout():
    from tests.test_reference import ChainSNN

    ir = parse_snn = None
    from spikeduet.frontend.snntorch_parser import parse_snntorch_model

    ir = parse_snntorch_model(ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0]))
    assert len(initialize_state(ir, 2)) == 2          # one membrane per LIF layer, no extension slots

def test_lowering_builds_delay_ring_state_buffers_and_ops():
    module = compile_source_to_execution_ir(tiny_ir(), num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING,
                                            pipeline=baseline_pipeline())
    names = {b.name: b for b in module.buffers}
    assert names["hist1"].shape == ((3 + 1) * 5,) and names["hist1"].stateful
    assert names["sprev0"].stateful and names["sprev1"].stateful and names["adapt1"].stateful
    assert any(isinstance(o, DelayPushOp) for o in module.loop[0].body)
    assert module.readout_weights is not None
    use = extension_usage(module)
    assert use.delays and use.spike_state and use.adaptation and use.weighted_readout

def test_delay_push_and_state_updates_are_never_hoisted_or_fused():
    ir = tiny_ir()
    base = compile_source_to_execution_ir(ir, num_timesteps=STEPS, input_temporal_mode=TemporalMode.INVARIANT,
                                          pipeline=baseline_pipeline())
    opt = compile_source_to_execution_ir(ir, num_timesteps=STEPS, input_temporal_mode=TemporalMode.INVARIANT,
                                         pipeline=optimized_pipeline())
    assert not any(isinstance(o, (DelayPushOp, LIFUpdateOp)) for o in opt.preheader_ops)
    # LICM hoists layer 0's affine (static input); fusion removes convert2 (LIF1 -> plain Linear2) but never
    # convert1, whose consumer is the delay ring rather than an affine.
    assert [o.id for o in opt.walk_ops() if o.id.startswith("convert")] == ["convert0", "convert1"]
    assert [o.id for o in base.walk_ops() if o.id.startswith("convert")] == ["convert0", "convert1", "convert2"]

def test_execution_verifier_checks_the_extension_buffers():
    import dataclasses

    module = compile_source_to_execution_ir(tiny_ir(), num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING,
                                            pipeline=baseline_pipeline())
    verify_execution_ir(module)
    bad_ops = tuple(dataclasses.replace(o, adapt=None) if isinstance(o, LIFUpdateOp) and o.adapt else o
                    for o in module.loop[0].body)
    with pytest.raises(ExecutionIRVerificationError):
        verify_execution_ir(dataclasses.replace(module, loop=(dataclasses.replace(module.loop[0], body=bad_ops),)))
    with pytest.raises(ExecutionIRVerificationError):
        verify_execution_ir(dataclasses.replace(module, readout_weights=(1.0, 2.0)))

@needs_cc
@pytest.mark.parametrize("pipeline", [baseline_pipeline, optimized_pipeline])
def test_portable_c_matches_reference_on_tiny_model(pipeline):
    ir = tiny_ir()
    x = tiny_inputs(120)
    art = compile_model(ir, backend="portable_c", num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING,
                        pipeline=pipeline(), include_harness=True)
    exe = compile_portable_c(art.files, Path(tempfile.mkdtemp()))
    got = np.array(run_harness(exe, x.reshape(len(x), -1).tolist()))
    assert (got == ref_scores(ir, x).argmax(1).numpy()).mean() >= 0.99

EFR32_DRIVER = r"""
#include "snn.c"
#include <stdio.h>
static float x[%d][%d];
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); snn_init();
    while (fread(x, sizeof(x), 1, f) == 1) {
        iscas_infer_reset(x[0]);
        for (int t = 0; t < %d; t++) iscas_infer_step(x[t]);
        printf("%%d", iscas_infer_readout());
        for (int j = 0; j < ISCAS_LAYERS[ISCAS_NUM_LAYERS - 1].out_features; j++) printf(" %%.9g", (double)g_mem_sum[j]);
        printf("\n");
    }
    return 0;
}
"""

def run_efr32_cpu_on_host(files, frames: np.ndarray, workdir: Path):
    """Host-compiles the generated EFR32 CPU engine (+ firmware/common/snn.h) and streams `frames` [N, T, C]."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model_data.h").write_text(files["model_data.h"])
    (workdir / "snn.c").write_text(files["snn_generated_cpu.c"])
    (workdir / "snn.h").write_text((REPO / "runtime" / "efr32" / "snn.h").read_text())
    frames.astype(np.float32).tofile(workdir / "x.bin")
    n, t, c = frames.shape
    (workdir / "main.c").write_text(EFR32_DRIVER % (t, c, t))
    exe = workdir / ("h.exe" if os.name == "nt" else "h")
    r = subprocess.run([CC, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-I", str(workdir), str(workdir / "main.c"), "-o", str(exe), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = subprocess.run([str(exe), str(workdir / "x.bin")], capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return np.array([int(r[0]) for r in rows]), np.array([[float(v) for v in r[1:]] for r in rows])

@needs_cc
@pytest.mark.parametrize("pipeline", [baseline_pipeline, optimized_pipeline])
def test_efr32_cpu_engine_matches_reference_scores_on_tiny_model(pipeline):
    ir = tiny_ir()
    x = tiny_inputs(80)
    module = compile_source_to_execution_ir(ir, num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING, pipeline=pipeline())
    files = Efr32Backend().generate(module, engine="cpu").files
    pred, scores = run_efr32_cpu_on_host(files, x.numpy(), Path(tempfile.mkdtemp()))
    ref = ref_scores(ir, x)
    assert np.abs(scores - ref.numpy()).max() < 1e-3
    assert (pred == ref.argmax(1).numpy()).all()

def test_extension_model_header_contains_extension_tables_and_ordinary_header_does_not():
    from tests.test_reference import ChainSNN
    from spikeduet.frontend.snntorch_parser import parse_snntorch_model

    module = compile_source_to_execution_ir(tiny_ir(), num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING)
    header = Efr32Backend().generate(module).files["model_data.h"]
    for token in ("ISCAS_LAYER_EXT", "iscas_layer1_synapse_delays", "iscas_layer1_alif_rho", "ISCAS_READOUT_WEIGHTS"):
        assert token in header
    plain = compile_source_to_execution_ir(
        parse_snntorch_model(ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])),
        num_timesteps=4, input_temporal_mode=TemporalMode.VARYING)
    plain_files = Efr32Backend().generate(plain, engine="cpu").files
    assert not any(tok in "".join(plain_files.values()) for tok in ("ISCAS_LAYER_EXT", "g_hist", "g_sprev", "g_adapt", "ISCAS_READOUT"))

def test_unsupported_realizations_refuse_extension_models():
    module = compile_source_to_execution_ir(tiny_ir(), num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING,
                                            pipeline=optimized_pipeline())
    with pytest.raises(ValueError, match="does not support the FilterBank extensions"):
        Efr32Backend().generate(module, engine="mvp")
    with pytest.raises(ValueError, match="does not support the FilterBank extensions"):
        Efr32Backend().generate(module, engine="cpu", sparse_cpu=True)
    with pytest.raises(ValueError, match="does not support the FilterBank extensions"):
        Efr32Backend().generate(module, engine="cpu", fold_output_affine=True)

def _tiny_module(ir=None, pipeline=None):
    return compile_source_to_execution_ir(ir or tiny_ir(), num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING,
                                          pipeline=pipeline or optimized_pipeline())

def test_bitmask_delay_history_only_for_provably_binary_producers():
    from spikeduet.backend.schedule_common import binary_spike_buffer, delay_bit_layers

    module = _tiny_module()
    assert delay_bit_layers(module) == {1: 3}
    assert binary_spike_buffer(module, "stage1") and binary_spike_buffer(module, "stage2")
    assert not binary_spike_buffer(module, "stage0")                      # the raw external input
    # graded (non-binary) spikes feeding the delay line keep the float ring
    ir = tiny_ir()
    ir.layers[1].graded_spikes_factor = [2.0]
    ir.layers[1].spike_state = False
    ir.layers[1].reset_delay = False
    assert delay_bit_layers(_tiny_module(ir)) == {}
    # a delay line fed directly by the external input has no binary producer either
    ir = tiny_ir()
    ir.layers[0].synapse_delays = torch.randint(0, 3, (5, 3))
    assert 0 not in delay_bit_layers(_tiny_module(ir))

@needs_cc
@pytest.mark.parametrize("pipeline", [baseline_pipeline, optimized_pipeline])
def test_speed_mode_and_bitmask_delays_are_bit_identical_to_the_plain_realization(pipeline):
    ir = tiny_ir()
    x = tiny_inputs(80, seed=5) * 1.5
    module = _tiny_module(ir, pipeline())
    plain = Efr32Backend().generate(module, engine="cpu", bitmask_delays=False).files
    fast = Efr32Backend().generate(module, engine="cpu", speed=True).files
    assert "g_hbits" not in plain["snn_generated_cpu.c"] and "g_hbits" in fast["snn_generated_cpu.c"]
    assert fast["snn_generated_cpu.c"].startswith("// Generated") and '#pragma GCC optimize ("O2")' in fast["snn_generated_cpu.c"]
    p1, s1 = run_efr32_cpu_on_host(plain, x.numpy(), Path(tempfile.mkdtemp()))
    p2, s2 = run_efr32_cpu_on_host(fast, x.numpy(), Path(tempfile.mkdtemp()))
    assert (p1 == p2).all()
    assert np.array_equal(s1.astype(np.float32).view(np.uint32), s2.astype(np.float32).view(np.uint32))

@needs_cc
def test_large_delay_layer_uses_generic_loop_and_stays_bit_identical():
    """More synapses than the unroll limit: the generic (non-unrolled) bitmask gather must still match."""
    from spikeduet.backend.schedule_common import UNROLL_DELAY_SYNAPSE_LIMIT

    g = torch.Generator().manual_seed(3)
    n_in, n_hid = 40, 30                                             # 1200 synapses > 1024
    assert n_in * n_hid > UNROLL_DELAY_SYNAPSE_LIMIT
    ir = SNNModel(
        [LinearLayer("l0", 3, n_in, torch.randn(n_in, 3, generator=g), torch.randn(n_in, generator=g) * 0.1),
         LIFLayer("a", beta=[0.8], threshold=[0.3], reset_mechanism="subtract", reset_delay=True, spike_state=True),
         LinearLayer("l1", n_in, n_hid, torch.randn(n_hid, n_in, generator=g) * 0.4, torch.randn(n_hid, generator=g) * 0.1,
                     synapse_delays=torch.randint(0, 12, (n_hid, n_in), generator=g)),
         LIFLayer("b", beta=[0.7], threshold=[0.5], reset_mechanism="subtract", reset_delay=True, spike_state=True),
         LinearLayer("l2", n_hid, 3, torch.randn(3, n_hid, generator=g)),
         LIFLayer("c", beta=[0.9], threshold=[1.0], reset_mechanism="none")], num_steps=STEPS)
    module = compile_source_to_execution_ir(ir, num_timesteps=STEPS, input_temporal_mode=TemporalMode.VARYING)
    x = tiny_inputs(40, seed=9) * 1.5
    plain = Efr32Backend().generate(module, engine="cpu", bitmask_delays=False).files
    fast = Efr32Backend().generate(module, engine="cpu", speed=True).files
    assert "E->synapse_delays[k]" in fast["snn_generated_cpu.c"]
    a = run_efr32_cpu_on_host(plain, x.numpy(), Path(tempfile.mkdtemp()))
    b = run_efr32_cpu_on_host(fast, x.numpy(), Path(tempfile.mkdtemp()))
    assert np.array_equal(a[1].astype(np.float32).view(np.uint32), b[1].astype(np.float32).view(np.uint32))
    assert np.abs(b[1] - ref_scores(ir, x).numpy()).max() < 1e-3

def test_speed_option_is_off_by_default_and_cpu_mvp_only():
    module = _tiny_module()
    default = Efr32Backend().generate(module, engine="cpu").files["snn_generated_cpu.c"]
    assert "#pragma GCC optimize" not in default and "spike-gated: binary input" not in default
    with pytest.raises(ValueError, match="engine='cpu' or 'mvp'"):
        Efr32Backend().generate(module, speed=True)                     # no engine

def _static_ir(seed=0, n0=24):
    g = torch.Generator().manual_seed(seed)
    r = lambda *sh: torch.randn(*sh, generator=g)
    return SNNModel(
        [LinearLayer("l0", 6, n0, r(n0, 6) * 0.9, r(n0) * 0.2),
         LIFLayer("a", beta=[0.9, 0.7, 0.5, 0.95] * (n0 // 4), threshold=[1.0], reset_mechanism="subtract", reset_delay=True),
         LinearLayer("l1", n0, 8, r(8, n0) * 0.5, r(8) * 0.1),
         LIFLayer("b", beta=[0.8], threshold=[1.0], reset_mechanism="subtract", reset_delay=True),
         LinearLayer("l2", 8, 3, r(3, 8), r(3) * 0.1),
         LIFLayer("c", beta=[0.9, 0.95, 0.85], threshold=[1.0], reset_mechanism="none")], num_steps=8)

def test_silent_neuron_bound_never_misses_a_spike():
    """The elimination rule: constant current <= 0.999*threshold*(1-beta) => the neuron never fires."""
    ir = _static_ir()
    x = torch.randn(300, 6) * 1.5
    lin, lif = ir.layers[0], ir.layers[1]
    cur = x @ lin.weight.T + lin.bias
    bound = 0.999 * lif.threshold[0] * (1 - torch.tensor(lif.beta))
    silent = cur <= bound
    with torch.no_grad():
        _, _, tr = run_sequence(ir, [x] * 8)
    fired = torch.stack([t.spikes[0] for t in tr]).sum(0) > 0
    assert silent.any() and (~silent).any() and fired.any()
    assert not (silent & fired).any()

@needs_cc
def test_silent_neuron_elimination_is_bit_identical_and_only_for_invariant_current():
    from spikeduet.backend.schedule_common import silent_lif_plans

    ir = _static_ir()
    module = compile_source_to_execution_ir(ir, num_timesteps=8, input_temporal_mode=TemporalMode.INVARIANT, pipeline=optimized_pipeline())
    plans = silent_lif_plans(module)
    assert list(plans) == ["lif0"] and plans["lif0"].n == 24            # not lif1 (varying current), not the readout layer
    x = (torch.randn(80, 6) * 1.5).numpy()
    frames = np.repeat(x[:, None, :], 8, axis=1)
    plain = Efr32Backend().generate(module, engine="cpu").files
    fast = Efr32Backend().generate(module, engine="cpu", speed=True).files
    assert "g_act0" in fast["snn_generated_cpu.c"] and "g_act0" not in plain["snn_generated_cpu.c"]
    p1, s1 = run_efr32_cpu_on_host(plain, frames, Path(tempfile.mkdtemp()))
    p2, s2 = run_efr32_cpu_on_host(fast, frames, Path(tempfile.mkdtemp()))
    assert (p1 == p2).all() and np.array_equal(s1.astype(np.float32).view(np.uint32), s2.astype(np.float32).view(np.uint32))
    # time-varying input, or no LICM (current computed inside the loop): no plan
    assert silent_lif_plans(compile_source_to_execution_ir(ir, num_timesteps=8, input_temporal_mode=TemporalMode.VARYING,
                                                           pipeline=optimized_pipeline())) == {}
    assert silent_lif_plans(compile_source_to_execution_ir(ir, num_timesteps=8, input_temporal_mode=TemporalMode.INVARIANT,
                                                           pipeline=baseline_pipeline())) == {}

def test_mvp_speed_mode_emits_o2_specialised_lif_and_silent_list():
    module = compile_source_to_execution_ir(_static_ir(), num_timesteps=8, input_temporal_mode=TemporalMode.INVARIANT,
                                            pipeline=optimized_pipeline())
    fast = Efr32Backend().generate(module, engine="mvp", speed=True).files["snn_generated_mvp.c"]
    plain = Efr32Backend().generate(module, engine="mvp").files["snn_generated_mvp.c"]
    assert '#pragma GCC optimize ("O2")' in fast and "#pragma GCC optimize" not in plain
    assert "silent-neuron elimination" in fast and "(float16_t)0.0f" in fast and "g_mvp_direct_fp16_write_count" in fast
    assert "silent-neuron" not in plain

def test_exact_floats_round_trip_float32_and_default_is_unchanged():
    from spikeduet.backend.c_common import float_literal

    rng = np.random.default_rng(0)
    for v in np.concatenate([rng.standard_normal(500).astype(np.float32) * 1e-3, np.array([1.0, -2.0, 0.0, 3e-9, 12345.678], np.float32)]):
        lit = float_literal(float(v), exact=True)
        assert lit.endswith("f") and np.float32(float(lit[:-1])) == v
    assert float_literal(0.123456789) == "0.123457f"                        # legacy six-decimal convention
    module = _tiny_module()
    plain = Efr32Backend().generate(module).files["model_data.h"]
    exact = Efr32Backend().generate(module, exact_floats=True).files["model_data.h"]
    assert plain != exact and plain == Efr32Backend().generate(module).files["model_data.h"]
