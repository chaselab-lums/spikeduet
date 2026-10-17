"""Tests for spikeduet.target.platform -- the board-neutral
PlatformSpec/AcceleratorSpec objects and the concrete EFR32MG24 specs.

These are plain data-object sanity checks (frozen, expected fields
present) plus the guarantee: nothing in spikeduet.ir depends on
spikeduet.target.
"""

import spikeduet.ir.model as ir_model
import spikeduet.ir.verifier as ir_verifier
from spikeduet.target.platform import EFR32MG24_CPU, EFR32MG24_MVP, AcceleratorSpec, PlatformSpec


def test_platform_specs_are_frozen():
    import dataclasses

    assert dataclasses.is_dataclass(PlatformSpec)
    try:
        EFR32MG24_MVP.name = "changed"
        assert False, "PlatformSpec should be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_efr32mg24_flash_and_sram_match_datasheet():
    # EFR32MG24B310F1536IM48: 1536 KB flash, 256 KB RAM (Silicon Labs datasheet).
    assert EFR32MG24_MVP.flash_bytes == 1536 * 1024
    assert EFR32MG24_MVP.sram_bytes == 256 * 1024
    assert EFR32MG24_CPU.flash_bytes == 1536 * 1024
    assert EFR32MG24_CPU.sram_bytes == 256 * 1024


def test_mvp_variant_has_accelerator_cpu_variant_does_not():
    assert EFR32MG24_MVP.accelerator is not None
    assert isinstance(EFR32MG24_MVP.accelerator, AcceleratorSpec)
    assert "float16" in EFR32MG24_MVP.accelerator.supported_dtypes
    assert EFR32MG24_CPU.accelerator is None


def test_unestablished_constraints_are_unspecified_not_guessed():
    # No MVP alignment or dimension limit has been established: they must stay None, not a guess.
    assert EFR32MG24_MVP.alignment_bytes is None
    assert EFR32MG24_MVP.accelerator.alignment_bytes is None
    assert EFR32MG24_MVP.accelerator.matrix_limits is None


def test_cpu_supports_only_float32_mvp_supports_both():
    assert EFR32MG24_CPU.supported_numeric_types == ("float32",)
    assert "float32" in EFR32MG24_MVP.supported_numeric_types
    assert "float16" in EFR32MG24_MVP.supported_numeric_types


def test_source_ir_modules_do_not_import_target_platform():
    """A valid source IR must not become invalid merely because
    of a target's constraints -- enforced structurally by never importing
    spikeduet.target from spikeduet.ir."""
    assert not hasattr(ir_model, "PlatformSpec")
    assert not hasattr(ir_verifier, "PlatformSpec")
    assert "target" not in ir_model.__file__ and "spikeduet.target" not in open(ir_model.__file__).read()
    assert "spikeduet.target" not in open(ir_verifier.__file__).read()
