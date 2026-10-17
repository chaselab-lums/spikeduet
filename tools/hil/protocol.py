"""Host side of the board's UART protocol, plus Simplicity Commander flash/reset helpers.

The board announces ``MCU_READY`` after boot. Static builds then take one line of scaled integers
per inference and answer ``PRED:<class>:<infer_us>:<parse_us>:<total_us>``. Temporal builds use
``run_temporal()``: a ``BEGIN_TEMPORAL`` header, then one vector per timestep, each acknowledged
with ``ACK_T:<t>`` before the next is sent. Bounding every round trip to one timestep keeps byte
loss rare and localizes it when it happens. Any ``ERR:...`` line from the board raises
``HilProtocolError`` at once.

Right after boot the board's UART receiver is not always armed when ``MCU_READY`` is printed; a
sample sent immediately can lose its first byte. ``open_and_sync`` therefore waits briefly and
flushes the input buffer before returning.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

import serial

COMMANDER_EXE = os.environ.get(
    "SPIKEDUET_COMMANDER", r"C:\SiliconLabs\SimplicityStudio\v5\developer\adapter_packs\commander\commander.exe")
DEFAULT_DEVICE = "EFR32MG24B310F1536IM48"
DEFAULT_BAUD = 115200
DEFAULT_SETTLE_S = 0.25
DEFAULT_READY_TIMEOUT_S = 8.0
DEFAULT_REPLY_TIMEOUT_S = 5.0
DEFAULT_ACK_TIMEOUT_S = 5.0
SCALE_FACTOR = 10000


class HilTimeoutError(TimeoutError):
    """The host gave up waiting for a reply."""


class HilProtocolError(RuntimeError):
    """The board reported a fault (an ``ERR:...`` line, e.g. a UART byte timeout) or broke the
    expected message order. Recover the same way as from a timeout: send a fresh request."""


@dataclass
class PredReply:
    predicted_class: int
    infer_us: int
    parse_us: int
    total_us: int


def commander_flash(hex_path: str, device: str = DEFAULT_DEVICE, commander: str = COMMANDER_EXE) -> None:
    result = subprocess.run([commander, "flash", hex_path, "--device", device], capture_output=True, text=True)
    if "Flashing completed successfully" not in result.stdout:
        raise RuntimeError(f"commander flash failed for {hex_path}:\n{result.stdout}\n{result.stderr}")


def commander_reset(device: str = DEFAULT_DEVICE, commander: str = COMMANDER_EXE) -> None:
    subprocess.run([commander, "device", "reset", "--device", device], capture_output=True)


def open_and_sync(
    port: str,
    baud: int = DEFAULT_BAUD,
    device: str = DEFAULT_DEVICE,
    commander: str = COMMANDER_EXE,
    settle_s: float = DEFAULT_SETTLE_S,
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
) -> serial.Serial:
    """Opens ``port``, resets the board, waits for ``MCU_READY``, settles and flushes. The returned
    connection can take a sample immediately."""
    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()
    commander_reset(device, commander)

    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        if b"MCU_READY" in ser.readline():
            break
    else:
        ser.close()
        raise HilTimeoutError(f"MCU_READY not observed within {ready_timeout_s}s on {port}")

    time.sleep(settle_s)
    ser.reset_input_buffer()
    return ser


def send_sample(ser: serial.Serial, scaled_values: Sequence[int]) -> None:
    ser.write((" ".join(str(int(v)) for v in scaled_values) + "\n").encode("ascii"))
    ser.flush()


def read_pred(ser: serial.Serial, reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S) -> PredReply:
    """Waits for the ``PRED`` line, skipping report lines; raises on an ``ERR`` line."""
    deadline = time.time() + reply_timeout_s
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("PRED:"):
            _, cls, infer_us, parse_us, total_us = line.split(":")
            return PredReply(int(cls), int(infer_us), int(parse_us), int(total_us))
        if line.startswith("ERR:"):
            raise HilProtocolError(f"MCU reported: {line}")
    raise HilTimeoutError(f"no PRED reply within {reply_timeout_s}s")


def send_and_read(ser: serial.Serial, scaled_values: Sequence[int],
                  reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S) -> PredReply:
    send_sample(ser, scaled_values)
    return read_pred(ser, reply_timeout_s)


def scale_vector(values: Sequence[float], scale: int = SCALE_FACTOR) -> list:
    return [int(round(v * scale)) for v in values]


# ---- temporal protocol (builds with ISCAS_INPUT_TEMPORAL) ----

def _read_control_line(ser: serial.Serial, timeout_s: float) -> str:
    """Returns the next non-empty line; raises ``HilProtocolError`` on ``ERR`` and
    ``HilTimeoutError`` if nothing arrives within ``timeout_s``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("ERR:"):
            raise HilProtocolError(f"MCU reported: {line}")
        return line
    raise HilTimeoutError(f"no response within {timeout_s}s")


def _expect_control_line(ser: serial.Serial, expected: str, timeout_s: float) -> None:
    """The next line must be exactly ``expected``; a wrong, stale or early line is an error."""
    line = _read_control_line(ser, timeout_s)
    if line != expected:
        raise HilProtocolError(f"expected {expected!r}, got {line!r}")


def send_temporal_header(ser: serial.Serial, num_steps: int, input_dim: int) -> None:
    """``BEGIN_TEMPORAL <steps> <dim>``; the board checks both numbers against its build."""
    ser.write(f"BEGIN_TEMPORAL {int(num_steps)} {int(input_dim)}\n".encode("ascii"))
    ser.flush()


def run_temporal(
    ser: serial.Serial,
    scaled_frames: Sequence[Sequence[int]],
    reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S,
    ack_timeout_s: float = DEFAULT_ACK_TIMEOUT_S,
    settle_s: float = 0.1,
) -> PredReply:
    """One temporal inference; ``scaled_frames`` holds one ``scale_vector()``-scaled vector per step.

        host -> BEGIN_TEMPORAL <steps> <dim>     board -> READY_T:0 (or ERR:BAD_HEADER)
        host -> <step t values>                   board -> ACK_T:<t>      for every step
                                                  board -> PRED:<class>:<infer_us>:<parse_us>:<total_us>

    ``ack_timeout_s`` bounds each READY_T/ACK_T wait, ``reply_timeout_s`` the final PRED wait.
    ``settle_s`` plus a flush before the header drops report lines the previous request may still
    be printing after its PRED. After any exception the board is ready for a fresh request.
    """
    num_steps = len(scaled_frames)
    input_dim = len(scaled_frames[0]) if scaled_frames else 0

    time.sleep(settle_s)
    ser.reset_input_buffer()

    send_temporal_header(ser, num_steps, input_dim)
    _expect_control_line(ser, "READY_T:0", ack_timeout_s)
    for t, frame in enumerate(scaled_frames):
        send_sample(ser, frame)
        _expect_control_line(ser, f"ACK_T:{t}", ack_timeout_s)
    return read_pred(ser, reply_timeout_s)
