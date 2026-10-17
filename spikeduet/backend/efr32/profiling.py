"""Opt-in per-stage DWT cycle markers for generated EFR32 engines."""

from typing import List, Set

from spikeduet.backend.schedule_common import layer_index_of
from spikeduet.ir.execution import AffineOp, ConvertOp, LIFUpdateOp


class StageProfile:
    """Per-stage DWT cycle attribution (``stage_profile=True``).

    Emits ``ISCAS_PROF_START()`` / ``ISCAS_PROF_LAP(i)``; a lap adds the cycles since the previous
    marker to stage ``i``. The markers compile to nothing unless the build defines
    ``ISCAS_STAGE_PROFILE``. With ``enabled=False`` nothing is emitted."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.names: List[str] = []
        self.substaged: Set[int] = set()     # affine layers whose emitter placed its own sub-stage laps

    def lap(self, name: str, indent: str = "    ") -> List[str]:
        """Idempotent by name, because the schedule is rendered more than once."""
        if not self.enabled:
            return []
        if name not in self.names:
            self.names.append(name)
        return [f"{indent}ISCAS_PROF_LAP({self.names.index(name)});  /* stage: {name} */"]

    def start(self, indent: str = "    ") -> List[str]:
        return [f"{indent}ISCAS_PROF_START();"] if self.enabled else []

    def epilogue(self, op) -> List[str]:
        if isinstance(op, ConvertOp):
            return self.lap(f"convert{layer_index_of(op.output)}")
        if isinstance(op, AffineOp):
            idx = layer_index_of(op.output)
            return [] if idx in self.substaged else self.lap(f"affine{idx}")
        if isinstance(op, LIFUpdateOp):
            return self.lap(f"lif{layer_index_of(op.current)}")
        return []

    def declarations(self) -> List[str]:
        if not self.enabled:
            return []
        return [
            "#ifdef ISCAS_STAGE_PROFILE",
            '#include "em_device.h"',
            "#define ISCAS_PROF_MAX 64",
            "static uint32_t g_prof_cycles[ISCAS_PROF_MAX];",
            "static uint32_t g_prof_t;",
            "#define ISCAS_PROF_START() (g_prof_t = DWT->CYCCNT)",
            "#define ISCAS_PROF_LAP(s) do { uint32_t _n = DWT->CYCCNT; g_prof_cycles[(s)] += _n - g_prof_t; g_prof_t = _n; } while (0)",
            "#else",
            "#define ISCAS_PROF_START() ((void)0)",
            "#define ISCAS_PROF_LAP(s) ((void)0)",
            "#endif",
        ]

    def reset_lines(self) -> List[str]:
        if not self.enabled:
            return []
        return ["#ifdef ISCAS_STAGE_PROFILE", "    memset(g_prof_cycles, 0, sizeof(g_prof_cycles));", "#endif"]

    def accessor_lines(self) -> List[str]:
        if not self.enabled:
            return []
        assert len(self.names) <= 64
        names = ", ".join(f'"{n}"' for n in self.names)
        return [
            "#ifdef ISCAS_STAGE_PROFILE",
            f"static const char *const g_prof_names[{len(self.names)}] = {{{names}}};",
            f"int iscas_debug_prof_count(void) {{ return {len(self.names)}; }}",
            "const char *iscas_debug_prof_name(int s) { return g_prof_names[s]; }",
            "unsigned long iscas_debug_prof_cycles(int s) { return (unsigned long)g_prof_cycles[s]; }",
            "#endif",
        ]


NO_PROFILE = StageProfile(False)
