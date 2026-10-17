"""Board runners on top of ``protocol``: flash a hex, sync once, run a list of inputs."""
from __future__ import annotations

import time
from typing import Optional, Sequence

from tools.hil.protocol import (
    DEFAULT_DEVICE,
    HilProtocolError,
    HilTimeoutError,
    PredReply,
    commander_flash,
    open_and_sync,
    run_temporal,
    send_and_read,
)


def run_temporal_vectors(
    hex_path: str,
    scaled_sequences: Sequence[Sequence[Sequence[int]]],
    port: str,
    device: str = DEFAULT_DEVICE,
    drain_timeout_s: float = 0.4,
    **run_temporal_kwargs,
) -> list:
    """Flashes ``hex_path``, syncs once and runs every ``[timestep][input_dim]`` scaled sequence.

    Returns one dict per sequence: ``reply`` (``PredReply`` or ``None``), ``extra`` (report lines the
    firmware printed after PRED, e.g. ``SPARSECOUNT:``, drained before the next request), ``error``
    and ``wall_s``.
    """
    commander_flash(hex_path, device=device)
    time.sleep(1.5)
    ser = open_and_sync(port, device=device, settle_s=1.0)
    results: list = []
    try:
        for seq in scaled_sequences:
            t0 = time.time()
            rec = {"reply": None, "extra": [], "error": None}
            try:
                rec["reply"] = run_temporal(ser, seq, **run_temporal_kwargs)
            except (HilTimeoutError, HilProtocolError) as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            old_timeout, ser.timeout = ser.timeout, drain_timeout_s
            try:
                while True:
                    raw = ser.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        rec["extra"].append(line)
            finally:
                ser.timeout = old_timeout
            rec["wall_s"] = round(time.time() - t0, 3)
            results.append(rec)
    finally:
        ser.close()
    return results


def run_vectors(
    hex_path: str,
    scaled_vectors: Sequence[Sequence[int]],
    port: str,
    device: str = DEFAULT_DEVICE,
    per_sample_timeout_s: float = 5.0,
) -> list:
    """Static builds: flashes ``hex_path``, syncs once and sends every vector. Returns one
    ``PredReply`` per vector, ``None`` where a sample timed out."""
    commander_flash(hex_path, device=device)
    ser = open_and_sync(port, device=device)

    results: list = []
    try:
        for vec in scaled_vectors:
            try:
                results.append(send_and_read(ser, vec, reply_timeout_s=per_sample_timeout_s))
            except HilTimeoutError:
                results.append(None)
    finally:
        ser.close()

    return results


def run_repeated_stability_check(
    hex_path: str,
    scaled_vector: Sequence[int],
    port: str,
    device: str = DEFAULT_DEVICE,
    num_boot_cycles: int = 20,
    samples_per_cycle: int = 5,
) -> dict:
    """Boot-sync stability check: ``num_boot_cycles`` reset+sync cycles of ``samples_per_cycle``
    requests each; reports how many requests timed out."""
    commander_flash(hex_path, device=device)

    total = 0
    timeouts = 0
    infer_us_values = []

    for _ in range(num_boot_cycles):
        ser = open_and_sync(port, device=device)
        try:
            for _ in range(samples_per_cycle):
                total += 1
                try:
                    reply: Optional[PredReply] = send_and_read(ser, scaled_vector)
                    infer_us_values.append(reply.infer_us)
                except HilTimeoutError:
                    timeouts += 1
        finally:
            ser.close()

    return {
        "total_requests": total,
        "timeouts": timeouts,
        "infer_us_values": infer_us_values,
    }
