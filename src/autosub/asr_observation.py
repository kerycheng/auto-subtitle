"""Preserve decoded text even when optional SDK observations cannot be recorded.

These are raw observations, not calibrated confidence or acoustic timestamps.
No tensor library is imported and no model or subtitle is modified.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real


def _plain(value, budget, depth=0):
    if depth > 20:
        raise ValueError("Observation nesting limit exceeded")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("Observation size limit exceeded")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite observation")
        return number
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Observation keys must be strings")
        return {key: _plain(item, budget, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, budget, depth + 1) for item in value]
    # Trusted SDK tensors/arrays only. Check the size before copying dense data.
    numel = getattr(value, "numel", None)
    size = numel() if callable(numel) else getattr(value, "size", None)
    if not isinstance(size, Integral) or size < 0 or size > budget[0]:
        raise ValueError("Unsupported or oversized SDK observation")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise ValueError("Unsupported SDK observation")
    return _plain(tolist(), budget, depth + 1)


def snapshot_hypothesis(hypothesis):
    """Extract text first, then isolate optional attribute/conversion failures.

    Missing optional fields are omitted, never synthesized or interpolated.
    SDK-specific timestamp/timestep names are retained with unknown units.
    Callers should persist this record before doing downstream analysis.
    """
    try:
        text = hypothesis.text
    except Exception as exc:
        raise ValueError("SDK result has no readable decoded text") from exc
    if not isinstance(text, str):
        raise ValueError("SDK decoded text must be a string")
    record = {"text": text, "observations": {}, "observation_errors": {},
              "timestamp_units": "SDK-native; not validated acoustic times"}
    for name in ("score", "y_sequence", "timestamp", "timestep", "tokens", "token_duration"):
        try:
            value = getattr(hypothesis, name)
        except AttributeError:
            continue
        except Exception as exc:
            record["observation_errors"][name] = type(exc).__name__
            continue
        try:
            record["observations"][name] = _plain(value, [10000])
        except Exception as exc:
            record["observation_errors"][name] = type(exc).__name__
    return record
