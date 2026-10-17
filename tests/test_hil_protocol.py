"""Host-side protocol tests, using a fake serial transport (no real
hardware needed) -- see docs/architecture/hil_protocol.md and hil/
protocol.py::run_temporal() for the full wire-protocol reference.

FakeSerial simulates just enough of pyserial's Serial interface for
hil/protocol.py's blocking-readline-with-timeout style: a pre-programmed
queue of lines `readline()` pops one at a time, returning b"" (pyserial's
own "nothing arrived within its configured timeout" convention) once the
queue is empty -- letting run_temporal()'s/read_pred()'s own
time.time()-based deadlines fire naturally. Every test uses a small
explicit ack_timeout_s/reply_timeout_s so a deliberately-triggered timeout
test takes a fraction of a second, not the multi-second hardware-scale
defaults.
"""
import pytest

from tools.hil.protocol import (
    HilProtocolError,
    HilTimeoutError,
    PredReply,
    run_temporal,
    scale_vector,
    send_temporal_header,
)


class FakeSerial:
    """Minimal fake of pyserial's Serial, backed by a fixed queue of lines
    to hand back from readline(). Records every write() call (undecoded)
    for assertions where useful."""

    def __init__(self, lines):
        self._lines = [line.encode("ascii") if isinstance(line, str) else line for line in lines]
        self.written = []

    def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        # run_temporal()'s pre-request settle+flush -- see its docstring.
        # A real reset_input_buffer() would discard unread bytes; nothing
        # to discard here since FakeSerial's queue only ever holds
        # intentionally-programmed lines for the test to consume.
        pass


def _frames(n=3, dim=4):
    return [scale_vector([0.1 * (t + 1)] * dim) for t in range(n)]


def test_ready_t0_accepted_and_full_happy_path():
    """READY_T:0 accepted; correct ACK_T sequence accepted; final PRED
    returned."""
    ser = FakeSerial([
        "READY_T:0",
        "ACK_T:0",
        "ACK_T:1",
        "ACK_T:2",
        "PRED:5:1000:200:1200",
    ])
    reply = run_temporal(ser, _frames(3), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)
    assert reply == PredReply(5, 1000, 200, 1200)

    # header was actually sent, with the right steps/dim
    assert ser.written[0] == b"BEGIN_TEMPORAL 3 4\n"
    # one write per timestep after the header
    assert len(ser.written) == 1 + 3


def test_wrong_timestep_ack_rejected():
    """ACK_T:<t> arriving out of order must raise HilProtocolError, not be
    silently accepted."""
    ser = FakeSerial([
        "READY_T:0",
        "ACK_T:0",
        "ACK_T:2",  # should have been ACK_T:1
    ])
    with pytest.raises(HilProtocolError, match="ACK_T:1"):
        run_temporal(ser, _frames(3), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)


def test_missing_ack_times_out():
    """No further lines after READY_T:0 -- the first ACK wait must raise
    HilTimeoutError, not hang."""
    ser = FakeSerial(["READY_T:0"])
    with pytest.raises(HilTimeoutError):
        run_temporal(ser, _frames(2), ack_timeout_s=0.2, reply_timeout_s=0.2, settle_s=0)


def test_err_uart_timeout_raises_immediately_not_after_full_timeout():
    """An ERR: line must raise HilProtocolError right away -- verified here
    by using a LARGE ack_timeout_s but expecting the raise regardless
    (FakeSerial hands back the ERR line on the very first readline(), so a
    correct implementation never actually waits out the timeout)."""
    ser = FakeSerial([
        "READY_T:0",
        "ACK_T:0",
        "ERR:UART_TIMEOUT:t=1:i=50",
    ])
    with pytest.raises(HilProtocolError, match="UART_TIMEOUT"):
        run_temporal(ser, _frames(3), ack_timeout_s=10.0, reply_timeout_s=10.0, settle_s=0)


def test_bad_header_error_raised():
    """A steps/dim mismatch reported by the MCU (ERR:BAD_HEADER) must
    raise HilProtocolError before any timestep is ever sent."""
    ser = FakeSerial(["ERR:BAD_HEADER:got_steps=3:got_dim=4:expected_steps=100:expected_dim=700"])
    with pytest.raises(HilProtocolError, match="BAD_HEADER"):
        run_temporal(ser, _frames(3), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)
    # only the header was ever written -- no timestep payload sent after a bad header
    assert len(ser.written) == 1


def test_pred_before_final_ack_rejected():
    """A PRED line arriving in place of an intermediate ACK_T must be
    treated as a protocol violation, not accepted as if it were the final
    answer."""
    ser = FakeSerial([
        "READY_T:0",
        "ACK_T:0",
        "PRED:1:100:50:150",  # premature -- t=1's ACK was expected
    ])
    with pytest.raises(HilProtocolError, match="ACK_T:1"):
        run_temporal(ser, _frames(3), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)


def test_extra_stale_ack_rejected():
    """A duplicate/stale ACK_T for an already-acknowledged timestep must
    be rejected (not silently consumed as if it were the next expected
    ACK)."""
    ser = FakeSerial([
        "READY_T:0",
        "ACK_T:0",
        "ACK_T:0",  # stale duplicate -- ACK_T:1 was expected
    ])
    with pytest.raises(HilProtocolError, match="ACK_T:1"):
        run_temporal(ser, _frames(3), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)


def test_send_temporal_header_format():
    ser = FakeSerial([])
    send_temporal_header(ser, 100, 700)
    assert ser.written == [b"BEGIN_TEMPORAL 100 700\n"]


def test_empty_lines_are_skipped_not_treated_as_protocol_violations():
    """Blank lines (e.g. stray '\\r\\n' framing noise) between real control
    lines must be skipped, not treated as an unexpected/garbage line."""
    ser = FakeSerial([
        "",
        "READY_T:0",
        "",
        "ACK_T:0",
        "PRED:2:10:5:15",
    ])
    reply = run_temporal(ser, _frames(1), ack_timeout_s=0.3, reply_timeout_s=0.3, settle_s=0)
    assert reply.predicted_class == 2
