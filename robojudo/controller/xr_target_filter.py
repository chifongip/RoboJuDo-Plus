"""Deterministic XR target filter; keep both repository copies identical."""

import math

import numpy as np


def filter_targets(desired, previous, limits, speeds, dt, alpha):
    if not math.isfinite(dt) or dt <= 0 or not 0 <= alpha < 1:
        raise ValueError("invalid filter settings")
    result = {}
    if set(desired) != set(previous):
        raise ValueError("incomplete desired targets")
    for name, value in desired.items():
        low, high = limits[name]
        old = previous[name]
        speed = speeds[name]
        if not all(math.isfinite(v) for v in (value, old, low, high, speed)) or speed < 0 or low > high:
            raise ValueError("invalid target filter input")
        value = min(high, max(low, value))
        filtered = alpha * old + (1 - alpha) * value
        if abs(filtered - value) < 0.001:
            filtered = value
        lower, upper = max(low, old - speed * dt), min(high, old + speed * dt)
        lo32, hi32 = np.float32(lower), np.float32(upper)
        if lo32 < lower:
            lo32 = np.nextafter(lo32, np.float32(np.inf))
        if hi32 > upper:
            hi32 = np.nextafter(hi32, np.float32(-np.inf))
        if lo32 > hi32 or not np.isfinite([lo32, hi32]).all():
            raise ValueError("no float32 target satisfies position and velocity limits")
        result[name] = float(np.clip(np.float32(filtered), lo32, hi32))
    return result
