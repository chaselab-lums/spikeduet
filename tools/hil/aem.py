"""Energy measurement with a Silicon Labs Advanced Energy Monitor (AEM), via Simplicity Commander.

AEM properties (measured on an EFM32PG28 Pro Kit): fixed 6,250 Hz sampling (160 us), each sample a
~160 us boxcar integral, ~0.79 uA current resolution at mA currents, ~23 uA per-sample noise, and
timestamps with no link to the MCU clock. A single inference cannot be isolated from UART traffic,
so energy is measured by replay: firmware built with ``ISCAS_ENERGY_REPLAY`` re-runs the same
inference ``reps`` times between two 15 ms EM1 gaps, and ``locate_burst`` finds those gaps from
their known spacing.

    E_inference = (E_burst('R') - E_burst('U')) / reps        # 'U' = the same loop without the inference
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from tools.hil.protocol import (COMMANDER_EXE, HilProtocolError, HilTimeoutError, commander_flash, open_and_sync,
                               run_temporal, scale_vector, send_and_read)

AEM_RATE_HZ = 6250.0
AEM_DT_US = 160.0
PG28_PART = "EFM32PG28B310F1024IM68"
AEM_JLINK_SERIAL = os.environ.get("SPIKEDUET_AEM_SERIAL")   # J-Link serial of the AEM kit, if several are attached
GAP_MS = 15.0                      # must match ISCAS_REPLAY_GAP_MS in runtime/efr32/app.c


# ---------------------------------------------------------------------------------------------------------
# AEM dump
# ---------------------------------------------------------------------------------------------------------
def start_dump(path: Path, duration_s: float, serialno: Optional[str] = AEM_JLINK_SERIAL) -> subprocess.Popen:
    cmd = [COMMANDER_EXE, "aem", "dump", "--duration", f"{duration_s:.2f}", "-o", str(path)]
    if serialno:
        cmd += ["-s", serialno]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_dump(path: Path):
    """Returns (t_us, i_mA, v_V) float arrays from a Commander `aem dump` CSV."""
    a = np.genfromtxt(str(path), delimiter=",", skip_header=1)
    return a[:, 0], a[:, 1], a[:, 2]


def _moving_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w          # element k = mean of x[k : k + w]


def locate_burst(t_us: np.ndarray, i_mA: np.ndarray, v_V: np.ndarray, burst_s: float, gap_ms: float = GAP_MS,
                 guard_ms: float = 1.5) -> Dict[str, float]:
    """Finds the replay burst in an AEM trace. The burst is flanked by two `gap_ms` EM1 gaps whose centres are
    exactly `burst_s + gap_ms` apart (burst_s from the MCU's DWT count), so the pair is the position that
    minimises (moving mean at a) + (moving mean at a + spacing): no current threshold, which matters because a
    dense-MVP burst can itself sit close to the EM1 level while the SDK waits for the MVP.
    Returns mean power inside the burst (guard_ms trimmed at each edge), the EM1 gap level and the polling level
    before the burst (UART wait, CPU running) for context."""
    dt = float(np.median(np.diff(t_us)))
    w = int(round(gap_ms * 1000.0 / dt)) - 8                 # gap window, trimmed against edge samples
    sp = int(round((burst_s * 1e6 + gap_ms * 1e3) / dt))     # centre-to-centre spacing in samples
    ma = _moving_mean(i_mA, w)                               # ma[k] ~ mean of i[k : k+w], centred at k + w/2
    n = len(ma) - sp
    if n < 10:
        raise ValueError("AEM trace shorter than the expected burst")
    score = ma[:n] + ma[sp:sp + n]
    a = int(np.argmin(score))
    # refine +-3 samples on the spacing (timestamps are 159/160 us, DWT vs AEM clocks differ slightly)
    best = (score[a], a, sp)
    for d in range(-3, 4):
        s2 = sp + d
        if a + s2 >= len(ma):
            continue
        sc = ma[a] + ma[a + s2]
        if sc < best[0]:
            best = (sc, a, s2)
    _, a, sp = best
    half = int(round(gap_ms * 1000.0 / dt / 2.0))
    c1, c2 = a + w // 2, a + sp + w // 2
    g = int(round(guard_ms * 1000.0 / dt))
    lo, hi = c1 + half + g, c2 - half - g
    p = i_mA[lo:hi] * v_V[lo:hi]                             # mW
    pre = i_mA[max(0, c1 - half - int(0.4e6 / dt)):max(1, c1 - half - 5)]
    return {
        "dt_us": dt, "c1": c1, "c2": c2, "spacing_s": (c2 - c1) * dt * 1e-6,
        "burst_samples": int(hi - lo), "p_burst_mW": float(p.mean()), "p_burst_std_mW": float(p.std()),
        "i_burst_mA": float(i_mA[lo:hi].mean()), "v_V": float(v_V[lo:hi].mean()),
        "i_gap_mA": float(0.5 * (ma[a] + ma[a + sp])), "i_poll_mA": float(np.median(pre)) if len(pre) else float("nan"),
        "t_burst_start_us": float(t_us[c1 + half]),
    }


# ---------------------------------------------------------------------------------------------------------
# Replay session
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ReplayResult:
    cmd: str
    reps: int
    burst_s: float
    clock_hz: int
    located: Dict[str, float]
    e_burst_mJ: float          # mean power x DWT burst time
    done_line: str = ""


@dataclass
class EnergyRequest:
    """One request (sample) of one build: the normal PRED reply plus its replay measurements."""
    pred: Optional[object] = None          # PredReply
    extras: List[str] = field(default_factory=list)
    reps: int = 0
    ref_cycles: int = 0
    runs: Dict[str, ReplayResult] = field(default_factory=dict)

    def energy_per_inference_mJ(self) -> Optional[float]:
        r, u = self.runs.get("R"), self.runs.get("U")
        if r is None:
            return None
        return (r.e_burst_mJ - (u.e_burst_mJ if u else 0.0)) / r.reps


class EnergySession:
    """Flashes `hex_path`, opens the VCOM, then per sample: one normal request (PRED + report lines) followed
    by AEM-captured replays. One J-Link serves flash, VCOM and AEM; they are used strictly one after another
    except that the AEM dump runs while the serial command is sent."""

    def __init__(self, hex_path: str, port: str, device: str = PG28_PART,
                 serialno: Optional[str] = AEM_JLINK_SERIAL, workdir: Optional[Path] = None):
        self.hex_path, self.port, self.device, self.serialno = hex_path, port, device, serialno
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="aem_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ser = None

    def open(self, flash: bool = True) -> None:
        if flash:
            commander_flash(self.hex_path, device=self.device)
            time.sleep(1.5)
        self.ser = open_and_sync(self.port, device=self.device, settle_s=1.0)

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.write(b"Q")
            except Exception:
                pass
            self.ser.close()
            self.ser = None

    # -- requests ------------------------------------------------------------------------------------
    def _read_until_replay_wait(self, req: EnergyRequest, timeout_s: float = 8.0) -> None:
        t0 = time.time()
        old, self.ser.timeout = self.ser.timeout, 0.5
        try:
            while time.time() - t0 < timeout_s:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
                if not line:
                    continue
                if line.startswith("REPLAY_WAIT"):
                    fields = dict(kv.split("=") for kv in line.split(":")[1:])
                    req.reps, req.ref_cycles = int(fields["reps"]), int(fields["ref_cycles"])
                    return
                req.extras.append(line)
        finally:
            self.ser.timeout = old
        raise HilTimeoutError("no REPLAY_WAIT from the firmware (not an ISCAS_ENERGY_REPLAY build?)")

    def request_temporal(self, frames: Sequence[Sequence[float]]) -> EnergyRequest:
        req = EnergyRequest()
        scaled = [scale_vector([float(x) for x in f]) for f in frames]
        req.pred = run_temporal(self.ser, scaled, ack_timeout_s=5.0, reply_timeout_s=10.0)
        self._read_until_replay_wait(req)
        return req

    def request_static(self, vector: Sequence[float]) -> EnergyRequest:
        req = EnergyRequest()
        req.pred = send_and_read(self.ser, scale_vector([float(x) for x in vector]))
        self._read_until_replay_wait(req)
        return req

    # -- replay ----------------------------------------------------------------------------------------
    def replay(self, req: EnergyRequest, cmd: str) -> ReplayResult:
        assert cmd in ("R", "U")
        # expected burst length: reps x reference inference (+ unpack overhead, bounded generously)
        clk = 39_000_000
        est = req.reps * req.ref_cycles / clk
        dur = est * 1.6 + 4.0
        csv = self.workdir / f"aem_{cmd}_{int(time.time() * 1000)}.csv"
        proc = start_dump(csv, dur, self.serialno)
        try:
            t0 = time.time()
            while time.time() - t0 < 20.0:                   # commander flushes in 6,250-sample (1 s) batches
                if csv.exists() and csv.stat().st_size > 100_000:
                    break
                time.sleep(0.1)
            else:
                raise HilTimeoutError("AEM dump never started")
            self.ser.reset_input_buffer()
            self.ser.write(cmd.encode())
            line = ""
            old, self.ser.timeout = self.ser.timeout, 1.0
            t1 = time.time()
            try:
                while time.time() - t1 < est * 1.6 + 10.0:
                    raw = self.ser.readline()
                    s = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
                    if s.startswith("REPLAY_ERR"):
                        raise HilProtocolError(s)
                    if s.startswith("REPLAY_DONE"):
                        line = s
                        break
                else:
                    raise HilTimeoutError("no REPLAY_DONE")
            finally:
                self.ser.timeout = old
            proc.wait(timeout=dur + 30.0)
        finally:
            if proc.poll() is None:
                proc.kill()
        f = line.split(":")
        kv = dict(x.split("=") for x in f[2:])
        reps, cyc, hz = int(kv["reps"]), int(kv["burst_cycles"]), int(kv["clock_hz"])
        t, i, v = load_dump(csv)
        loc = locate_burst(t, i, v, cyc / hz)
        return ReplayResult(cmd=cmd, reps=reps, burst_s=cyc / hz, clock_hz=hz, located=loc,
                            e_burst_mJ=loc["p_burst_mW"] * (cyc / hz), done_line=line)

    def measure(self, req: EnergyRequest, with_unpack: bool = True) -> EnergyRequest:
        req.runs["R"] = self.replay(req, "R")
        if with_unpack:
            req.runs["U"] = self.replay(req, "U")
        return req
