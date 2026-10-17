"""Tests for spikeduet.backend.api -- explicit backend selection, and
that the existing EFR32 output (spikeduet.backend.model_header.generate_c_model) is
unchanged/reachable through the new Backend interface.
"""

import pytest

from spikeduet.backend.api import compile_model, get_backend
from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.portable_c import PortableCBackend
from spikeduet.backend.model_header import generate_c_model
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from tests.test_reference import ChainSNN


def _source_ir():
    model = ChainSNN(dims=[(4, 3), (3, 2)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    return parse_snntorch_model(model)


def test_get_backend_returns_correct_types():
    assert isinstance(get_backend("portable_c"), PortableCBackend)
    assert isinstance(get_backend("efr32"), Efr32Backend)


def test_get_backend_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("mvp_only_backend_that_does_not_exist")


def test_compile_model_portable_c_produces_expected_files():
    artifact = compile_model(_source_ir(), backend="portable_c", num_timesteps=5)
    assert set(artifact.files) == {
        "model_data.h", "model_config.h", "portable_runtime.h", "portable_runtime.c",
    }


def test_compile_model_portable_c_with_harness():
    artifact = compile_model(_source_ir(), backend="portable_c", num_timesteps=5, include_harness=True)
    assert "main.c" in artifact.files


def test_compile_model_efr32_matches_existing_codegen_output():
    ir = _source_ir()
    artifact = compile_model(ir, backend="efr32", num_timesteps=5)
    direct = generate_c_model(ir)

    assert artifact.files["model_data.h"] == direct


def test_compile_model_requires_explicit_backend_no_hidden_default():
    with pytest.raises(TypeError):
        compile_model(_source_ir())  # backend is keyword-only, no default


def test_input_temporal_mode_is_forwarded():
    from spikeduet.ir.execution import TemporalMode as TM

    artifact_invariant = compile_model(
        _source_ir(), backend="portable_c", num_timesteps=5, input_temporal_mode=TM.INVARIANT
    )
    artifact_varying = compile_model(
        _source_ir(), backend="portable_c", num_timesteps=5, input_temporal_mode=TM.VARYING
    )

    assert "ISCAS_INPUT_TEMPORAL 0" in artifact_invariant.files["model_config.h"]
    assert "ISCAS_INPUT_TEMPORAL 1" in artifact_varying.files["model_config.h"]
