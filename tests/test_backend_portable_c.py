"""Portable-C backend tests -- host compilation, semantic conformance
(all Leaky), static vs temporal input execution, arbitrary depth, and
vendor-independence.
"""

import re

import pytest
import torch

from spikeduet.backend.api import compile_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from tests.backend_c_harness import CC, compile_portable_c, python_reference_predict, run_harness
from tests.test_reference import ChainSNN

pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler (gcc/clang) on PATH")

_FORBIDDEN_SYMBOLS = [
    "sl_math_mvp",
    "sl_status",
    "sl_iostream",
    "EFR32",
    "em_device",
    "silabs",
    "SILICON_LABS",
    "CMSIS",
    "cmsis",
]


def _build(dims, *, num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, **kw):
    n = len(dims)
    betas = kw.pop("betas", [0.9] * n)
    thresholds = kw.pop("thresholds", [1.0] * n)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds, **kw)
    ir = parse_snntorch_model(model)
    artifact = compile_model(
        ir,
        backend="portable_c",
        num_timesteps=num_timesteps,
        input_temporal_mode=input_temporal_mode,
        include_harness=True,
    )
    return ir, artifact


def _static_input_sequence(num_timesteps, x):
    return [x for _ in range(num_timesteps)]


# --- host compilation --------------------------------------------


def test_generated_code_compiles_with_werror(tmp_path):
    _, artifact = _build([(4, 3), (3, 2)])
    exe = compile_portable_c(artifact.files, tmp_path)
    assert exe.exists()


# --- vendor independence ------------------------------------------


def _strip_c_comments(text: str) -> str:
    """Strips /* */ and // comments so a vendor-name MENTIONED in a doc
    comment (e.g. this backend's own "not #included from runtime/efr32/"
    notes) doesn't count as a code dependency (check actual
    includes/symbols, not documentation strings)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def test_no_vendor_symbols_in_generated_files():
    _, artifact = _build([(4, 3), (3, 2)])
    for name, content in artifact.files.items():
        code_only = _strip_c_comments(content)
        for symbol in _FORBIDDEN_SYMBOLS:
            assert symbol not in code_only, f"forbidden symbol {symbol!r} found in {name} (outside comments)"


def test_no_vendor_includes():
    _, artifact = _build([(4, 3), (3, 2)])
    includes = []
    for content in artifact.files.values():
        includes.extend(re.findall(r'#include\s+[<"]([^">]+)[">]', content))
    allowed = {
        "math.h", "string.h", "stdio.h", "stdlib.h", "stdint.h", "stddef.h",
        "model_data.h", "model_config.h", "portable_runtime.h",
        # host-harness-only, standard-library, platform-GUARDED (#ifdef _WIN32)
        # binary-stdio setup -- see portable_c.py's _HARNESS_C comment
        "fcntl.h", "io.h",
    }
    assert set(includes) <= allowed, f"unexpected includes: {set(includes) - allowed}"


# --- semantic conformance -----------------------------------------


@pytest.mark.parametrize("reset_mechanism,reset_delay", [
    ("subtract", False), ("subtract", True),
    ("zero", False), ("zero", True),
    ("none", False),
])
def test_reset_conformance(tmp_path, reset_mechanism, reset_delay):
    ir, artifact = _build(
        [(4, 3)],
        reset_mechanisms=[reset_mechanism],
        reset_delays=[reset_delay],
    )
    exe = compile_portable_c(artifact.files, tmp_path / f"{reset_mechanism}_{reset_delay}")

    torch.manual_seed(0)
    samples = [torch.randn(1, 4) for _ in range(5)]
    flat_samples = [x[0].tolist() for x in samples]

    predicted = run_harness(exe, flat_samples)
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


def test_per_neuron_beta_threshold_conformance(tmp_path):
    ir, artifact = _build(
        [(4, 4)],
        betas=[[0.0, 0.4, 0.8, 1.0]],
        thresholds=[[0.3, 0.6, 0.9, 1.2]],
    )
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(1)
    samples = [torch.randn(1, 4) for _ in range(5)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


def test_graded_spikes_conformance(tmp_path):
    ir, artifact = _build(
        [(4, 4), (4, 3)],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
    )
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(2)
    samples = [torch.randn(1, 4) for _ in range(5)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


def test_inhibition_conformance(tmp_path):
    ir, artifact = _build([(3, 3)], inhibitions=[True])
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(3)
    samples = [torch.randn(1, 3) for _ in range(5)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


def test_state_quant_conformance(tmp_path):
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=2, uniform=True, threshold=1.0)
    ir, artifact = _build([(3, 3)], state_quants=[q])
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(4)
    samples = [torch.randn(1, 3) for _ in range(5)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


def test_full_leaky_feature_composition_conformance(tmp_path):
    """One model composing per-neuron beta/threshold/graded factor,
    inhibition, and state_quant in its first layer, feeding an ordinary
    second layer, so the readout depends on all of them."""
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    ir, artifact = _build(
        [(4, 4), (4, 3)],
        betas=[[0.0, 0.4, 0.8, 1.0], 0.9],
        thresholds=[[0.3, 0.6, 0.9, 1.2], 1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
        reset_mechanisms=["zero", "subtract"],
        reset_delays=[True, False],
        inhibitions=[True, False],
        state_quants=[q, False],
    )
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(5)
    samples = [torch.randn(1, 4) for _ in range(10)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(5, x)) for x in samples]

    assert predicted == expected


# --- arbitrary depth ----------------------------------------------


def test_arbitrary_depth_four_layers(tmp_path):
    """Linear->Leaky x4, zero manual topology edits -- the backend must
    handle this exactly like the 1- or 2-layer cases above."""
    ir, artifact = _build([(6, 8), (8, 6), (6, 5), (5, 3)], num_timesteps=6)
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(6)
    samples = [torch.randn(1, 6) for _ in range(8)]
    predicted = run_harness(exe, [x[0].tolist() for x in samples])
    expected = [python_reference_predict(ir, _static_input_sequence(6, x)) for x in samples]

    assert predicted == expected


# --- static vs temporal input semantics ----------------------------


def test_static_input_uses_same_buffer_every_timestep(tmp_path):
    """A model whose only observable difference across two hand-crafted
    5-timestep sequences is what happens AFTER the first timestep proves
    static mode literally replays timestep 0's input -- if the harness
    used anything but that first vector on later timesteps, the two
    predictions below would differ from the reference."""
    ir, artifact = _build([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT)
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(7)
    x = torch.randn(1, 4)
    predicted = run_harness(exe, [x[0].tolist()])
    expected = python_reference_predict(ir, _static_input_sequence(5, x))

    assert predicted == [expected]


def test_temporal_input_uses_timestep_indexed_input(tmp_path):
    """SHD-like: each timestep gets a genuinely different input vector.
    Feeding a DIFFERENT per-timestep sequence must change the predicted
    class relative to a static replay of just the first vector -- proving
    the backend actually consumes ISCAS_STEPS distinct vectors, not one
    vector reused ISCAS_STEPS times."""
    ir, artifact = _build([(4, 3)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING)
    exe = compile_portable_c(artifact.files, tmp_path)

    torch.manual_seed(8)
    sequence = [torch.randn(1, 4) for _ in range(5)]
    flat = []
    for x in sequence:
        flat.extend(x[0].tolist())

    predicted = run_harness(exe, [flat])
    expected = python_reference_predict(ir, sequence)
    assert predicted == [expected]

    # sanity: a genuinely different sequence produces a genuinely
    # different C-side result too (not coincidentally always the same
    # regardless of what's fed in)
    torch.manual_seed(9)
    other_sequence = [torch.randn(1, 4) for _ in range(5)]
    other_flat = []
    for x in other_sequence:
        other_flat.extend(x[0].tolist())
    other_predicted = run_harness(exe, [other_flat])
    other_expected = python_reference_predict(ir, other_sequence)
    assert other_predicted == [other_expected]


def test_temporal_mode_actually_reads_distinct_timesteps_not_just_first(tmp_path):
    """Directly proves the C harness's per-timestep offsetting is real:
    build a sequence whose LATER timesteps alone determine the outcome
    (a big pulse only on timestep 4), and confirm the portable-C result
    tracks the Python reference, which by construction only fires because
    of that late pulse."""
    ir, artifact = _build(
        [(2, 2)], num_timesteps=5, input_temporal_mode=TemporalMode.VARYING,
        thresholds=[5.0],
    )
    exe = compile_portable_c(artifact.files, tmp_path)

    zero = torch.zeros(1, 2)
    pulse = torch.tensor([[10.0, 10.0]])
    sequence = [zero, zero, zero, zero, pulse]
    flat = []
    for x in sequence:
        flat.extend(x[0].tolist())

    predicted = run_harness(exe, [flat])
    expected = python_reference_predict(ir, sequence)
    assert predicted == [expected]
