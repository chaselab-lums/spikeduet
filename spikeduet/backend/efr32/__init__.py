"""Silicon Labs EFR32 backend (Cortex-M33 + MVP matrix accelerator).

``Efr32Backend.generate()`` is the entry point; see ``codegen.py`` for every option.
"""

from spikeduet.backend.efr32.codegen import Efr32Backend, generate_efr32_cpu_schedule, generate_efr32_mvp_schedule
from spikeduet.backend.efr32.fold import output_fold_plan
from spikeduet.backend.efr32.placement import (
    ActivityProfile,
    PlacementError,
    format_placement_report,
    plan_engine_placement,
    plan_with_reordering,
    profile_activity,
    profile_reordering,
)

__all__ = [
    "ActivityProfile",
    "Efr32Backend",
    "PlacementError",
    "format_placement_report",
    "generate_efr32_cpu_schedule",
    "generate_efr32_mvp_schedule",
    "output_fold_plan",
    "plan_engine_placement",
    "plan_with_reordering",
    "profile_activity",
    "profile_reordering",
]
