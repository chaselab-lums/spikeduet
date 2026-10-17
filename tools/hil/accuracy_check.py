"""Prediction agreement and latency statistics for board runs.

Agreement compares two prediction sources (e.g. the reference simulator and the board); it is not
classification accuracy against labels."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from tools.hil.protocol import PredReply


def prediction_agreement(expected: Sequence[int], actual: Sequence[Optional[PredReply]]) -> dict:
    """Compares expected class indices against HIL replies (None = timeout,
    counted as a mismatch, not silently dropped).
    """
    n = len(expected)
    matches = 0
    mismatches = []
    for i, (exp, act) in enumerate(zip(expected, actual)):
        got = act.predicted_class if act is not None else None
        if got == exp:
            matches += 1
        else:
            mismatches.append({"index": i, "expected": exp, "actual": got})
    return {"n": n, "matches": matches, "mismatches": mismatches}


def latency_stats(replies: Sequence[Optional[PredReply]], field: str = "infer_us") -> dict:
    values = [getattr(r, field) for r in replies if r is not None]
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(arr),
        "avg": float(arr.mean()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
