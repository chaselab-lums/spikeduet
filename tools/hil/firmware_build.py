"""Build automation for Simplicity Studio projects (slc-cli + GNU Arm + make).

A firmware variant is a copy of an existing working project in which only the generated sources
and some ``ISCAS_*`` defines change; the source project is never modified.

    clone_project("my_mvp_project", "my_mvp_sparse",
                  files={"snn.c": code, "model_data.h": header},
                  add_defines=["ISCAS_REPORT_RAWCYCLES=1"])
    build("my_mvp_sparse")                   # raises on an error or any compiler warning
    size("my_mvp_sparse"), static_sram("my_mvp_sparse"), requested_heap("my_mvp_sparse")

Toolchain locations come from the ``SPIKEDUET_*`` environment variables below.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

# Toolchain locations. Override with environment variables; the defaults are the Simplicity Studio 5
# install paths on Windows.
WORKSPACE = Path(os.environ.get("SPIKEDUET_SSV5_WORKSPACE", Path.home() / "SimplicityStudio" / "v5_workspace"))
ARM_GCC_DIR = os.environ.get(
    "SPIKEDUET_ARM_GCC_DIR", "C:/SiliconLabs/SimplicityStudio/v5/developer/toolchains/gnu_arm/12.2.rel1_2023.7")
SDK_PATH = os.environ.get("SPIKEDUET_SDK_PATH", str(Path.home() / "SimplicityStudio" / "SDKs" / "simplicity_sdk"))
EXTRA_PATH = [p for p in os.environ.get("SPIKEDUET_BUILD_PATH", "").split(os.pathsep) if p]   # mingw32-make, sed, uname
REPO_APP_C = Path(__file__).resolve().parents[2] / "runtime" / "efr32" / "app.c"


def project_dir(name: str) -> Path:
    return WORKSPACE / name


def _makefile(d: Path) -> Path:
    mk = list(d.glob("*.Makefile"))
    assert len(mk) == 1, f"expected one *.Makefile in {d}, found {mk}"
    return mk[0]


def clone_project(src: str, dst: str, files: Optional[Dict[str, str]] = None,
                  add_defines: Iterable[str] = (), remove_defines: Iterable[str] = (),
                  refresh_app_c: bool = True, overwrite: bool = True) -> Path:
    """Copies project ``src`` to ``dst`` (without build output), renames the Makefile, project.mak and
    PROJECTNAME, writes ``files`` (name -> text), refreshes app.c from ``runtime/efr32`` and edits the
    ``-D`` defines."""
    s, d = project_dir(src), project_dir(dst)
    assert s.is_dir(), s
    if d.exists():
        if not overwrite:
            raise FileExistsError(d)
        shutil.rmtree(d)
    shutil.copytree(s, d, ignore=shutil.ignore_patterns("build", "build.log"))
    mk = _makefile(d)
    old = mk.name[: -len(".Makefile")]
    text = mk.read_text()
    text = text.replace(f"PROJECTNAME = {old}", f"PROJECTNAME = {dst}").replace(f"{old}.project.mak", f"{dst}.project.mak")
    mk.unlink()
    (d / f"{dst}.Makefile").write_text(text)
    pm_old, pm_new = d / f"{old}.project.mak", d / f"{dst}.project.mak"
    pm_old.rename(pm_new)
    pm = pm_new.read_text()
    for define in remove_defines:
        pm = "\n".join(l for l in pm.splitlines() if f"-D{define}" not in l) + "\n"
    anchor = " '-DDEBUG_EFM=1' \\\n"
    assert pm.count(anchor) >= 1, "C_DEFS anchor not found"
    for define in add_defines:
        pm = pm.replace(anchor, anchor + f" '-D{define}' \\\n")
    pm_new.write_text(pm)
    for name, content in (files or {}).items():
        (d / name).write_text(content)
    if refresh_app_c:
        shutil.copyfile(REPO_APP_C, d / "app.c")
    return d


SLT_ARCHIVE = Path(os.environ.get("SPIKEDUET_SLT_ARCHIVE", Path.home() / ".silabs" / "slt" / "installs" / "archive"))
SLC_ENV_DIRS = [str(SLT_ARCHIVE / "slc-cli-v6.0.23" / "slc_cli"), str(SLT_ARCHIVE / "python-v3.10.3"),
                str(SLT_ARCHIVE / "java21-v21.0.6" / "bin")]
PG28_PART = "EFM32PG28B310F1024IM68"      # PG28 Pro Kit PK2506A (BRD2506A): AEM-equipped second board
PG28_BOARD = "brd2506a"
PG28_FLASH_BYTES = 1024 * 1024
MG24_PART = "EFR32MG24B310F1536IM48"
MG24_BOARD = "brd2601b"


def port_project(src: str, dst: Optional[str] = None, part: str = PG28_PART, board: str = PG28_BOARD,
                 add_defines: Iterable[str] = (), jobs: int = 8, build_it: bool = True,
                 configuration: Optional[Dict[str, str]] = None) -> Path:
    """Ports a project to another Series-2 part and board (default: the EFM32PG28 Pro Kit), leaving the
    source project untouched.

    The top-level .c/.h sources (except slc's main.c) are copied; app.c comes from ``runtime/efr32``; the .slcp gets the new part and board ids,
    every `-DISCAS_*` define of the source's project.mak (clone_project() only edits project.mak, never the slcp)
    as slcp defines, plus `add_defines`; SL_BOARD_ENABLE_VCOM=1 is set because the Pro Kit gates its VCOM link
    with a board-controller pin (PD6) and without it the UART TX is garbled after the first line. The project is
    regenerated with slc and (build_it) built with the standard flow. A build that overflows the part's flash
    raises RuntimeError from build(); callers record that as "does not fit".
    """
    dst = dst or f"PG28_{src}"
    s, d = project_dir(src), project_dir(dst)
    assert s.is_dir(), s
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in s.iterdir():
        if f.is_file() and f.suffix in (".c", ".h", ".md") and f.name != "main.c":
            shutil.copyfile(f, d / f.name)
    shutil.copyfile(REPO_APP_C, d / "app.c")

    slcps = list(s.glob("*.slcp"))          # clone_project() keeps the template's .slcp name
    assert len(slcps) == 1, f"expected one .slcp in {s}, found {slcps}"
    text = slcps[0].read_text()
    text = text.replace(f"project_name: {slcps[0].stem}", f"project_name: {src}").replace(f"label: {slcps[0].stem}", f"label: {src}")
    text = text.replace(f"project_name: {src}", f"project_name: {dst}").replace(f"label: {src}", f"label: {dst}")
    text = text.replace(MG24_PART, part).replace(MG24_BOARD, board)
    pms = list(s.glob("*.project.mak"))
    assert len(pms) == 1, f"expected one .project.mak in {s}, found {pms}"
    pm = pms[0].read_text()
    iscas_defs = []
    for dflag in re.findall(r"'-D(ISCAS_[A-Za-z0-9_]+(?:=[^']*)?)'", pm):
        if dflag not in iscas_defs:
            iscas_defs.append(dflag)
    for dflag in add_defines:
        if dflag not in iscas_defs:
            iscas_defs.append(dflag)
    # The source's project.mak is the truth for the ISCAS_* defines (clone_project() edits only project.mak, so the
    # slcp may still list defines the clone removed, e.g. ISCAS_COUNT_EXEC_GEN on mlp_reference builds): drop every
    # ISCAS_* slcp define and re-add exactly the project.mak set.
    text = "\n".join(l for l in text.splitlines() if not re.match(r"^- \{name: ISCAS_", l)) + "\n"
    lines = []
    for dflag in iscas_defs:
        name, _, value = dflag.partition("=")
        lines.append(f"- {{name: {name}, value: '{value}'}}" if value else f"- {{name: {name}}}")
    block = ""
    cfg = {"SL_BOARD_ENABLE_VCOM": "1", **(configuration or {})}
    if "configuration:" not in text:
        block += "configuration:\n" + "".join(f"- {{name: {k}, value: '{v}'}}\n" for k, v in cfg.items())
    if "define:" in text:
        text = text.replace("define:\n", block + "define:\n" + "".join(l + "\n" for l in lines), 1)
    else:
        text += "\n" + block + "define:\n" + "".join(l + "\n" for l in lines)
    (d / f"{dst}.slcp").write_text(text)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(SLC_ENV_DIRS + [env.get("PATH", "")])
    slc = shutil.which("slc", path=env["PATH"]) or "slc"      # CreateProcess does not search the child's PATH
    r = subprocess.run([slc, "generate", "-p", f"{dst}.slcp", "-s", SDK_PATH, "-d", ".", "-o", "makefile",
                        "-tlcn", "gcc", "--overwrite-all"], cwd=d, capture_output=True, text=True, env=env)
    if r.returncode != 0 or "has completed" not in r.stdout:
        raise RuntimeError(f"slc generate failed for {dst}:\n{(r.stdout + r.stderr)[-3000:]}")
    if build_it:
        build(dst, jobs=jobs)
    return d


def build(name: str, jobs: int = 8, allow_warnings: bool = False) -> str:
    """Builds with the standard CLI flow; returns the log. Raises on failure or on any warning."""
    d = project_dir(name)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([*EXTRA_PATH, env.get("PATH", "")])
    cmd = ["mingw32-make", f"-j{jobs}", "-f", f"{name}.Makefile", f"ARM_GCC_DIR={ARM_GCC_DIR}", f"SDK_PATH={SDK_PATH}"]
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    (d / "build.log").write_text(log)
    if r.returncode != 0:
        raise RuntimeError(f"build of {name} failed:\n{log[-4000:]}")
    warnings = [l for l in log.splitlines() if "warning:" in l]
    if warnings and not allow_warnings:
        raise RuntimeError(f"build of {name} produced {len(warnings)} warning(s):\n" + "\n".join(warnings[:20]))
    return log


def hex_path(name: str) -> str:
    return str(project_dir(name) / "build" / "debug" / f"{name}.hex")


def size(name: str) -> Dict[str, int]:
    """`arm-none-eabi-size` of the built .out: text/data/bss, and Flash = text + data."""
    out = project_dir(name) / "build" / "debug" / f"{name}.out"
    r = subprocess.run([f"{ARM_GCC_DIR}/bin/arm-none-eabi-size", str(out)], capture_output=True, text=True, check=True)
    text, data, bss = (int(x) for x in r.stdout.splitlines()[1].split()[:3])
    return {"text": text, "data": data, "bss": bss, "flash": text + data, "size_line": r.stdout.splitlines()[1]}


def static_sram(name: str) -> Dict[str, int]:
    """Static SRAM = .stack + .bss + .data from the linker map. Do not use ``size``'s bss column: it
    also contains the .heap section, which the Silicon Labs linker script stretches to the end of RAM."""
    m = (project_dir(name) / "build" / "debug" / f"{name}.map").read_text()
    sec = {}
    for s in ("stack", "bss", "data", "heap"):
        hit = re.search(rf"^\.{s}\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)", m, re.M)
        sec[s] = int(hit.group(1), 16) if hit else 0
    sec["static_total"] = sec["stack"] + sec["bss"] + sec["data"]
    return sec


# Host stand-in for the MVP SDK: an FP16 matvec, not bit-exact to the device. Used to host-compile generated
# MVP code for logic tests and for requested_heap().
MVP_HOST_STUB = """#ifndef SL_MATH_MVP_H
#define SL_MATH_MVP_H
typedef _Float16 float16_t;
typedef struct { int rows, cols; const float16_t *data; } sl_math_matrix_f16_t;
static inline void sl_math_matrix_init_f16(sl_math_matrix_f16_t *m, int r, int c, float16_t *d) { m->rows = r; m->cols = c; m->data = d; }
static inline int sl_math_mvp_matrix_vector_mult_f16(const sl_math_matrix_f16_t *m, const float16_t *x, float16_t *y)
{ for (int j = 0; j < m->rows; j++) { float a = 0.0f; for (int i = 0; i < m->cols; i++) a += (float)m->data[j * m->cols + i] * (float)x[i]; y[j] = (float16_t)a; } return 0; }
#endif
"""


def requested_heap(name: str, workdir: Optional[Path] = None) -> Dict[str, int]:
    """Exact heap requested by ``snn_init()``: host-compiles the project's own snn.c with a counting
    malloc/calloc and runs it once. Allocator overhead is not included (about 8 B per allocation). Uses the
    project's ISCAS_* defines except ISCAS_STAGE_PROFILE. Range-sparse MVP builds program MVP registers
    directly and cannot be host-compiled; use ``placement.estimate_sram`` for them."""
    import tempfile
    src = project_dir(name)
    code = (src / "snn.c").read_text()
    if "sl_mvp.h" in code or "iscas_sp_launch" in code:
        raise ValueError(f"{name}: range-sparse MVP code cannot be host-compiled; use placement.estimate_sram() instead")
    pm = (src / f"{name}.project.mak").read_text()
    defs = [d for d in re.findall(r"'-D(ISCAS_[A-Z0-9_]+(?:=[^']*)?)'", pm) if not d.startswith("ISCAS_STAGE_PROFILE")]
    d = Path(workdir or tempfile.mkdtemp(prefix=f"heap_{name}_"))
    d.mkdir(parents=True, exist_ok=True)
    for f in ("model_data.h", "snn.h"):
        shutil.copy(src / f, d / f)
    (d / "sl_math_mvp.h").write_text(MVP_HOST_STUB)
    shim = "\n".join(["#include <stdlib.h>", "static size_t g_heap_bytes, g_heap_allocs;",
                    "static void *cnt_malloc(size_t n) { g_heap_bytes += n; g_heap_allocs++; return malloc(n); }",
                    "static void *cnt_calloc(size_t a, size_t b) { g_heap_bytes += a * b; g_heap_allocs++; return calloc(a, b); }", ""])
    main = "\n".join(["", "#include <stdio.h>",
                    'int main(void) { snn_init(); printf("%zu %zu%c", g_heap_bytes, g_heap_allocs, 10); return 0; }', ""])
    (d / "h.c").write_text(shim + code.replace("malloc(", "cnt_malloc(").replace("calloc(", "cnt_calloc(") + main)
    cc = shutil.which("gcc") or shutil.which("clang")
    exe = d / ("h.exe" if os.name == "nt" else "h")
    r = subprocess.run([cc, "-std=gnu11", "-O0", "-w", "-I", str(d), *[f"-D{x}" for x in defs], str(d / "h.c"), "-o", str(exe)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{name}: host compile failed:\n{r.stderr[-2000:]}")
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    heap, n = map(int, out.strip().splitlines()[-1].split())        # last line: snn_init() may print check lines first
    return {"heap_requested": heap, "allocations": n, "alloc_overhead_est": 8 * n}
