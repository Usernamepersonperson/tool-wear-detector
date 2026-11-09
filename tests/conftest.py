from pathlib import Path

import numpy as np
import pytest

from toolwear import features
from toolwear.model import Table

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "cache" / "mill.mat"


def synth_signal(wear: float, rng: np.random.Generator, n: int = 3000) -> dict[str, np.ndarray]:
    """Fake cut: idle, a cut window with load and noise growing with `wear`, idle again."""
    sig = {}
    cut = np.zeros(n, dtype=bool)
    cut[n // 5 : 4 * n // 5] = True
    load = np.where(cut, 5.0 + 4.0 * wear, 1.5) + rng.normal(0, 0.05, n)
    sig["smcDC"] = load
    sig["smcAC"] = np.where(cut, 1.0 + wear, 0.1) * np.sin(np.arange(n) * 0.9) + rng.normal(0, 0.05, n)
    for ch in ("vib_table", "vib_spindle", "AE_table", "AE_spindle"):
        sig[ch] = 0.1 + np.where(cut, 0.5 + wear, 0.0) * np.abs(rng.normal(0, 1, n))
    return sig


def synth_table(n_cutters: int = 4, n_runs: int = 8, seed: int = 0) -> Table:
    rng = np.random.default_rng(seed)
    case, run, vb, time, proc, feats = [], [], [], [], [], []
    for c in range(1, n_cutters + 1):
        rate = 0.05 + 0.01 * c
        for j in range(n_runs):
            t = 3.0 * (j + 1)
            w = rate * t / 3.0
            case.append(c)
            run.append(j + 1)
            vb.append(w)
            time.append(t)
            proc.append([1.5, 0.25, 1 + c % 2])
            feats.append(features.run_features(synth_signal(w, rng)))
    return Table(
        case=np.array(case),
        run=np.array(run),
        vb=np.array(vb),
        time=np.array(time),
        process=np.array(proc, dtype=float),
        feats=np.array(feats),
    )


@pytest.fixture(scope="session")
def table() -> Table:
    return synth_table()


@pytest.fixture(scope="session")
def real_results():
    if not MAT.exists():
        pytest.skip("public data set not in cache/ (run: python -m toolwear fetch)")
    from toolwear import data, evaluate, model

    runs = data.load_runs(MAT)
    return evaluate.run_all(model.build_table(runs), runs)
