"""Wear smoothing and remaining-useful-life (RUL) estimation by trend extrapolation."""

from __future__ import annotations

import numpy as np

RUL_CAP = 120.0  # minutes; returned when the wear trend is flat or falling


def smooth_wear(pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Causal exponential smoothing followed by a running maximum (wear never recovers)."""
    p = np.asarray(pred, dtype=float)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty_like(p)
    for i, v in enumerate(p):
        out[i] = v if i == 0 else alpha * v + (1 - alpha) * out[i - 1]
    return np.maximum.accumulate(out) if out.size else out


def time_to_limit(times: np.ndarray, vb: np.ndarray, limit: float) -> float | None:
    """Cutting time at which measured VB first reaches `limit` (linear interpolation).

    Unmeasured (nan) cuts are skipped. None when the limit is never reached.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(vb, dtype=float)
    m = np.isfinite(v)
    t, v = t[m], v[m]
    for i in range(len(v)):
        if v[i] >= limit:
            if i == 0:
                return float(t[0])
            dv = v[i] - v[i - 1]
            if dv <= 0:
                return float(t[i])
            return float(t[i - 1] + (limit - v[i - 1]) / dv * (t[i] - t[i - 1]))
    return None


def estimate_rul(
    times: np.ndarray,
    wear: np.ndarray,
    limit: float,
    window: int = 6,
    min_points: int = 3,
    cap: float = RUL_CAP,
) -> float:
    """RUL in minutes from the last `window` smoothed wear values (linear trend to the limit).

    Returns nan with fewer than `min_points`, 0 when the limit is already reached and `cap` when
    the fitted slope is not positive.
    """
    t = np.asarray(times, dtype=float)
    w = np.asarray(wear, dtype=float)
    if t.shape != w.shape:
        raise ValueError("times and wear must have the same shape")
    if t.size < min_points:
        return float("nan")
    if w[-1] >= limit:
        return 0.0
    t, w = t[-window:], w[-window:]
    if np.ptp(t) <= 0:
        return float(cap)
    slope, intercept = np.polyfit(t, w, 1)
    if slope <= 1e-9:
        return float(cap)
    return float(min(cap, max(0.0, (limit - (slope * t[-1] + intercept)) / slope)))
