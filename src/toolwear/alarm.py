"""Alarms: severe-wear risk alarm and an impulse (breakage / chipping) detector."""

from __future__ import annotations

import numpy as np

from toolwear import features

# Dynamic-content features that jump when an edge chips: crest of vibration / AE / load, load level.
IMPULSE_FEATURES = (
    "vib_table_crest",
    "vib_spindle_crest",
    "AE_table_crest",
    "AE_spindle_crest",
    "smcAC_crest",
    "smcDC_mean_abs",
    "vib_table_kurt",
    "vib_spindle_kurt",
    "AE_table_kurt",
    "AE_spindle_kurt",
)
MAD_FLOOR = 0.05  # absolute floor of the robust scale
REL_FLOOR = 0.15  # plus this share of |median|, so unbounded features (kurtosis) are not over-weighted


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that the share of negatives with score >= t is <= fpr."""
    s = np.sort(np.asarray(neg_scores, dtype=float))
    if s.size == 0:
        raise ValueError("need at least one negative score")
    if not 0 <= fpr < 1:
        raise ValueError("fpr must be in [0, 1)")
    k = int(np.floor(fpr * s.size))  # negatives allowed above the threshold
    if k == 0:
        return float(np.nextafter(s[-1], np.inf))
    return float(np.nextafter(s[-k - 1], np.inf))


def recall_at(scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return float("nan")
    return float(np.mean(s >= threshold))


def false_alarm_rate(scores: np.ndarray, threshold: float) -> float:
    return recall_at(scores, threshold)


def first_alarm(times: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    """Cutting time of the first cut whose score reaches the threshold, else None."""
    t = np.asarray(times, dtype=float)
    s = np.asarray(scores, dtype=float)
    hit = np.where(s >= threshold)[0]
    return float(t[hit[0]]) if hit.size else None


def impulse_score(history: np.ndarray, current: np.ndarray, min_history: int = 3) -> float:
    """Largest robust z-score of the impulse features against the tool's own recent cuts.

    `history` is (k, n_features) of previous cuts of the same tool (all 42 features), `current`
    the new cut. Returns nan when fewer than `min_history` previous cuts exist.
    """
    h = np.asarray(history, dtype=float)
    c = np.asarray(current, dtype=float)
    if h.ndim != 2 or h.shape[0] < min_history:
        return float("nan")
    names = features.feature_names()
    cols = [names.index(n) for n in IMPULSE_FEATURES]
    hh, cc = h[:, cols].copy(), c[cols].copy()
    for k, n in enumerate(IMPULSE_FEATURES):
        if n.endswith("_kurt"):  # excess kurtosis is unbounded: compress it to a log scale
            hh[:, k] = np.log10(np.maximum(hh[:, k] + 3.0, 1e-3))
            cc[k] = np.log10(max(cc[k] + 3.0, 1e-3))
    med = np.median(hh, axis=0)
    mad = 1.4826 * np.median(np.abs(hh - med), axis=0)
    scale = np.maximum(mad, MAD_FLOOR + REL_FLOOR * np.abs(med))
    z = (cc - med) / scale
    return float(np.max(z))


def inject_breakage(
    sig: dict[str, np.ndarray], rng: np.random.Generator, burst: float = 6.0, step: float = 0.2
) -> dict[str, np.ndarray]:
    """Copy of a cut with a synthetic chipping event: decaying impulse burst in the vibration and
    AE channels and a step in spindle load. Used only to test the alarm; always labelled injected.
    """
    out = {k: np.array(v, dtype=float, copy=True) for k, v in sig.items()}
    w = features.cut_window(out["smcDC"])
    lo, hi = w.start, w.stop
    pos = int(rng.integers(lo, max(lo + 1, hi - 40)))
    n = min(40, out["smcDC"].size - pos)
    env = np.exp(-np.arange(n) / 8.0)
    for ch in ("vib_table", "vib_spindle", "AE_table", "AE_spindle"):
        x = out[ch]
        sd = float(np.std(x[w])) or 1.0
        x[pos : pos + n] += burst * sd * env * rng.choice([-1.0, 1.0], size=n)
    load = out["smcDC"]
    idle = float(np.median(load[: max(1, load.size // 20)]))
    load[pos:] += step * (float(np.max(load)) - idle)
    return out
