"""Tool-change policies: fixed tool-life counter versus wear-aware recommendation."""

from __future__ import annotations

import numpy as np


def counter_life(train_lives: list[float], mode: str = "conservative") -> float:
    """Fixed tool-life counter (minutes of cutting) set from the lives of known tools.

    conservative: the shortest known life (no known tool would have been over-run).
    mean: the average known life.
    """
    lives = [float(x) for x in train_lives]
    if not lives:
        raise ValueError("need at least one tool life")
    if mode == "conservative":
        return min(lives)
    if mode == "mean":
        return float(np.mean(lives))
    raise ValueError(f"unknown counter mode {mode!r}")


def recipe_counter(
    known: list[tuple[tuple, float]], recipe: tuple, factor: float = 1.0
) -> float:
    """Per-recipe tool-life counter: mean life of known tools run with the same recipe
    (material, depth of cut, feed), scaled by a safety `factor`. Falls back to the mean life of
    all known tools when the recipe has not been seen.
    """
    if not known:
        raise ValueError("need at least one known tool")
    if factor <= 0:
        raise ValueError("factor must be positive")
    same = [life for rec, life in known if rec == recipe]
    base = float(np.mean(same)) if same else float(np.mean([life for _, life in known]))
    return factor * base


def recommend_change(times: np.ndarray, wear: np.ndarray, trigger: float) -> int | None:
    """Index of the first cut whose smoothed wear reaches the trigger, else None."""
    w = np.asarray(wear, dtype=float)
    hit = np.where(w >= trigger)[0]
    return int(hit[0]) if hit.size else None


def change_time(times: np.ndarray, wear: np.ndarray, trigger: float) -> tuple[float, bool]:
    """Cutting time at which the wear-aware policy changes the tool, and whether it was observed.

    When the trigger never fires inside the log, the last logged time is returned with False.
    """
    t = np.asarray(times, dtype=float)
    i = recommend_change(t, wear, trigger)
    if i is None:
        return float(t[-1]), False
    return float(t[i]), True


def outcome(change_at: float, true_life: float, part_min: float) -> dict[str, float]:
    """Result of changing a tool at `change_at` when its real end of life is `true_life`."""
    if part_min <= 0:
        raise ValueError("part_min must be positive")
    late = change_at > true_life
    return {
        "cut_min": change_at,
        "parts": change_at / part_min,
        "late": float(late),
        "overrun_min": max(0.0, change_at - true_life),
        "unused_min": max(0.0, true_life - change_at),
    }
