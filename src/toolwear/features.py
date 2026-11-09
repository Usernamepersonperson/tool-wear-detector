"""Explainable signal features: RMS, kurtosis, crest factor, spectral band energy."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import stats

FS = 250.0  # Hz, sampling rate of the public milling data set
CHANNELS = ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle")
BANDS = ((1.0, 40.0), (40.0, 80.0), (80.0, 124.0))  # Hz
STATS = ("mean_abs", "rms", "kurt", "crest", "band0", "band1", "band2")
EPS = 1e-12


def feature_names(channels=CHANNELS) -> list[str]:
    return [f"{c}_{s}" for c in channels for s in STATS]


def band_energy(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Power in [lo, hi) Hz from a Welch periodogram (sum of PSD bins times bin width)."""
    x = np.asarray(x, dtype=float)
    if x.size < 16:
        raise ValueError("signal too short for a spectrum")
    f, p = sps.welch(x - x.mean(), fs=fs, nperseg=min(256, x.size))
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0
    return float(np.sum(p[m]) * (f[1] - f[0]))


def channel_features(x: np.ndarray, fs: float = FS) -> dict[str, float]:
    """Seven features for one channel. Positive-valued ones are returned in log10."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 16 or not np.all(np.isfinite(x)):
        raise ValueError("signal must have >=16 finite samples")
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(ac**2)))
    peak = float(np.max(np.abs(ac)))
    out = {
        "mean_abs": float(np.log10(np.mean(np.abs(x)) + EPS)),
        "rms": float(np.log10(rms + EPS)),
        "kurt": float(stats.kurtosis(ac)) if rms > 0 else 0.0,
        "crest": float(np.log10(peak / (rms + EPS) + EPS)),
    }
    for i, (lo, hi) in enumerate(BANDS):
        out[f"band{i}"] = float(np.log10(band_energy(x, fs, lo, hi) + EPS))
    return out


def cut_window(load: np.ndarray, trim: float = 0.1, min_len: int = 64) -> slice:
    """Steady in-cut window from the spindle-load channel.

    The cut is where the load is above the midpoint between the idle level (median of the first
    5 percent of samples) and its maximum. The first and last `trim` fraction of that span are
    dropped (entry/exit transients). Falls back to the whole record when no cut is found.
    """
    x = np.asarray(load, dtype=float).ravel()
    n = x.size
    if n < min_len:
        return slice(0, n)
    idle = float(np.median(x[: max(1, n // 20)]))
    hi = float(np.max(x))
    if hi - idle < 1e-9:
        return slice(0, n)
    above = np.where(x > idle + 0.5 * (hi - idle))[0]
    if above.size < min_len:
        return slice(0, n)
    a, b = int(above[0]), int(above[-1]) + 1
    k = int((b - a) * trim)
    a, b = a + k, b - k
    return slice(a, b) if b - a >= min_len else slice(0, n)


def run_features(sig: dict[str, np.ndarray], fs: float = FS) -> np.ndarray:
    """Feature vector for one cut (steady in-cut window), ordered as feature_names()."""
    w = cut_window(sig["smcDC"])
    vals = []
    for c in CHANNELS:
        f = channel_features(np.asarray(sig[c], dtype=float)[w], fs)
        vals.extend(f[s] for s in STATS)
    return np.asarray(vals, dtype=float)


def relative_to_baseline(feats: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Delta vs the tool's first logged cut. All features are already log or unitless."""
    return np.asarray(feats, dtype=float) - np.asarray(baseline, dtype=float)


def power_trend(mean_abs_log: np.ndarray, window: int = 3) -> np.ndarray:
    """Spindle-load trend: trailing moving average of the (log) mean spindle current."""
    v = np.asarray(mean_abs_log, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.empty_like(v)
    for i in range(v.size):
        out[i] = v[max(0, i - window + 1) : i + 1].mean()
    return out
