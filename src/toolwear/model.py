"""Feature table and the one learned model (extra-trees regressor for flank wear VB)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from toolwear import features
from toolwear.data import Run

# Explainable sensor features fed to the model (chosen by leave-cutter-out CV on this data set).
SENSOR_FEATURES = (
    "smcAC_crest",
    "smcDC_rms",
    "smcDC_mean_abs",
    "vib_spindle_crest",
    "vib_table_crest",
    "AE_spindle_kurt",
    "AE_spindle_rms",
    "AE_table_rms",
    "vib_spindle_rms",
    "smcDC_band0",
)
VARIANTS = ("sensors+time", "time-only", "sensors-only")


@dataclass
class Table:
    """One row per cut, ordered by cutter then run."""

    case: np.ndarray
    run: np.ndarray
    vb: np.ndarray  # mm, nan when not measured
    time: np.ndarray  # cumulative cutting time, min
    process: np.ndarray  # (n, 3): depth of cut, feed, material
    feats: np.ndarray  # (n, 42) in features.feature_names() order

    def __len__(self) -> int:
        return len(self.case)

    def cutters(self) -> list[int]:
        return [int(c) for c in np.unique(self.case)]

    def rows(self, case: int) -> np.ndarray:
        idx = np.where(self.case == case)[0]
        return idx[np.argsort(self.time[idx], kind="stable")]

    def design(self, variant: str = "sensors+time") -> np.ndarray:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        names = features.feature_names()
        cols = [names.index(n) for n in SENSOR_FEATURES]
        parts = []
        if variant != "time-only":
            parts.append(self.feats[:, cols])
        if variant != "sensors-only":
            parts.append(self.time[:, None])
        parts.append(self.process)
        return np.hstack(parts)


def build_table(runs: list[Run], min_runs: int = 3) -> Table:
    """Feature table; cutters with fewer than `min_runs` cuts are dropped."""
    counts: dict[int, int] = {}
    for r in runs:
        counts[r.case] = counts.get(r.case, 0) + 1
    keep = [r for r in runs if counts[r.case] >= min_runs]
    if not keep:
        raise ValueError("no cutter has enough runs")
    keep.sort(key=lambda r: (r.case, r.time, r.run))
    return Table(
        case=np.array([r.case for r in keep]),
        run=np.array([r.run for r in keep]),
        vb=np.array([r.vb for r in keep], dtype=float),
        time=np.array([r.time for r in keep], dtype=float),
        process=np.array([[r.doc, r.feed, r.material] for r in keep], dtype=float),
        feats=np.array([features.run_features(r.signals) for r in keep]),
    )


class WearModel:
    """Flank-wear regressor. Predictions are clipped to be non-negative."""

    def __init__(self, variant: str = "sensors+time", seed: int = 0, n_trees: int = 300):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self._m = ExtraTreesRegressor(
            n_estimators=n_trees, min_samples_leaf=2, random_state=seed, n_jobs=1
        )

    def fit(self, table: Table, idx: np.ndarray) -> WearModel:
        idx = np.asarray(idx)
        idx = idx[np.isfinite(table.vb[idx])]
        if idx.size < 5:
            raise ValueError("need at least 5 labelled cuts to fit")
        self._m.fit(table.design(self.variant)[idx], table.vb[idx])
        return self

    def predict(self, table: Table, idx: np.ndarray) -> np.ndarray:
        x = table.design(self.variant)[np.asarray(idx)]
        return np.clip(self._m.predict(x), 0.0, None)

    def importances(self) -> dict[str, float]:
        """Feature importances by name, for explainability."""
        names = []
        if self.variant != "time-only":
            names += list(SENSOR_FEATURES)
        if self.variant != "sensors-only":
            names.append("cut_time")
        names += ["depth_of_cut", "feed", "material"]
        return dict(zip(names, map(float, self._m.feature_importances_), strict=True))
