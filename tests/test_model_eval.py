import json
from pathlib import Path

import numpy as np
import pytest

from toolwear import cli, data, evaluate, figures, model, report
from toolwear.data import Run


def _run(case, n, sig):
    return Run(case=case, run=n, vb=0.1, time=float(n), doc=1.0, feed=0.2, material=1, signals=sig)


def test_build_table_drops_short_cutters_and_sorts():
    from conftest import synth_signal

    rng = np.random.default_rng(0)
    runs = [_run(1, n, synth_signal(0.1 * n, rng, 600)) for n in (3, 1, 2)]
    runs.append(_run(2, 1, synth_signal(0.1, rng, 600)))
    t = model.build_table(runs)
    assert t.cutters() == [1]
    assert t.time.tolist() == [1.0, 2.0, 3.0]


def test_build_table_empty_raises():
    with pytest.raises(ValueError):
        model.build_table([])


def test_design_variants_have_expected_widths(table):
    n_s = len(model.SENSOR_FEATURES)
    assert table.design("sensors+time").shape[1] == n_s + 1 + 3
    assert table.design("time-only").shape[1] == 1 + 3
    assert table.design("sensors-only").shape[1] == n_s + 3
    with pytest.raises(ValueError):
        table.design("nope")


def test_wear_model_learns_synthetic_wear(table):
    train = np.where(table.case != 1)[0]
    test = np.where(table.case == 1)[0]
    m = model.WearModel(n_trees=100).fit(table, train)
    pred = m.predict(table, test)
    assert pred.min() >= 0
    assert evaluate.mae(pred, table.vb[test]) < 0.15


def test_wear_model_needs_labels(table):
    with pytest.raises(ValueError):
        model.WearModel().fit(table, np.arange(3))


def test_importances_cover_all_columns(table):
    m = model.WearModel(n_trees=20).fit(table, np.arange(len(table)))
    imp = m.importances()
    assert len(imp) == table.design().shape[1]
    assert sum(imp.values()) == pytest.approx(1.0)


def test_loco_never_trains_on_the_test_cutter(table, monkeypatch):
    seen = []
    orig = model.WearModel.fit

    def spy(self, tab, idx):
        seen.append(set(tab.case[np.asarray(idx)].tolist()))
        return orig(self, tab, idx)

    monkeypatch.setattr(model.WearModel, "fit", spy)
    pred = evaluate.loco_predict(table, "time-only")
    assert np.isfinite(pred).all()
    for held, trained in zip(table.cutters(), seen, strict=True):
        assert held not in trained


def test_mean_baseline_is_loco(table):
    p = evaluate.loco_mean_baseline(table)
    m = table.case == 1
    assert p[m][0] == pytest.approx(table.vb[~m].mean())


def test_mae_ignores_nan():
    assert evaluate.mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 5.0])) == 1.0
    assert np.isnan(evaluate.mae(np.array([1.0]), np.array([np.nan])))


def _series(life, rate=0.02, n=12):
    t = np.arange(1.0, n + 1) * 3.0
    vb = rate * t
    return {
        "recipe": (1.0, 1.0, 1.0),
        "rows": np.arange(n),
        "t": t,
        "vb": vb,
        "pred": vb,
        "smooth": vb,
        "life": life,
        "severe_t": None,
    }


def test_policy_with_perfect_wear_estimate_has_no_scrap():
    series = {1: _series(20.0), 2: _series(20.0, 0.02), 3: _series(10.0, 0.04)}
    for c, s in series.items():
        s["life"] = 0.4 / (0.02 if c < 3 else 0.04)
    res = evaluate.evaluate_policy(series, margin=0.1)
    assert res["summary"]["wear-aware"]["scrap_risk"] == 0.0
    assert set(res["summary"]) >= {"counter-global-min", "wear-aware"}


def test_rul_with_perfect_wear_is_accurate():
    s = _series(20.0)
    res = evaluate.evaluate_rul({1: s, 2: _series(20.0)})
    assert res["rul_mae_min"] < 1.5


def test_evaluate_alarm_on_perfect_scores():
    t = np.arange(1.0, 11.0)
    vb = np.linspace(0.05, 0.8, 10)
    s = {
        "t": t,
        "vb": vb,
        "smooth": vb,
        "severe_t": float(np.interp(0.6, vb, t)),
    }
    res = evaluate.evaluate_alarm(None, {1: s})
    assert res["run_recall"] == 1.0
    assert res["false_alarm_rate"] == 0.0


def test_report_and_figures(tmp_path):
    t = np.arange(1.0, 9.0) * 3
    ex = {
        "case": 1,
        "t": t.tolist(),
        "vb": (0.04 * t / 3).tolist(),
        "smooth": (0.04 * t / 3).tolist(),
        "true_life_min": 30.0,
        "alarm_at_min": None,
        "wear_aware_change_min": 18.0,
        "counter_global_min": 6.0,
        "counter_recipe_0.8x": 20.0,
    }
    row = {"mean_cut_min": 10.0, "mean_parts": 3.3, "scrap_risk": 0.1, "mean_unused_life_min": 2.0}
    keys = ["counter-global-min", "counter-recipe-0.6x", "counter-recipe-0.8x", "wear-aware"]
    res = {
        "n_cutters": 2, "n_cuts": 16, "n_labelled": 16, "limit_mm": 0.4, "severe_mm": 0.6,
        "wear_mae_mm": {"sensors+time": 0.1, "time-only": 0.1, "sensors-only": 0.12},
        "wear_mae_mean_baseline_mm": 0.2,
        "rul": {"n_points": 4, "rul_mae_min": 5.0, "counter_rul_mae_min": 6.0,
                "late_rul_mae_min": 4.0, "late_counter_rul_mae_min": 5.0},
        "alarm": {"threshold_mm": 0.5, "fpr_target": 0.05, "false_alarm_rate": 0.0,
                  "run_recall": 0.5, "tools_caught": 1, "tools_reaching_severe": 2,
                  "median_lead_min": 1.0},
        "policy": {"n_tools": 2, "margin_mm": 0.15, "part_min": 3.0,
                   "summary": {k: dict(row) for k in keys},
                   "wear_aware_vs": {"counter-global-min": {"extra_parts_per_tool": 1.0,
                                     "life_gain_pct": 10.0, "scrap_risk_delta": 0.0}}},
        "per_run": {"vb": (0.04 * t / 3).tolist(), "pred": (0.04 * t / 3).tolist()},
        "example": ex,
    }
    text = report.format_report(res)
    assert "0.100" in text and "wear-aware" in text
    from PIL import Image

    figures.make_figure(res, tmp_path / "f.png")
    figures.make_post_figure(res, tmp_path / "p.png")
    assert Image.open(tmp_path / "f.png").size == (1600, 900)
    assert Image.open(tmp_path / "p.png").size == (960, 700)


def test_cli_parser():
    p = cli.build_parser()
    assert p.parse_args(["demo", "--no-figures"]).no_figures is True
    assert p.parse_args(["fetch"]).cache == "cache"
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_md5_of_and_fetch_uses_existing(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert data.md5_of(f) == "900150983cd24fb0d6963f7d28e17f72"
    (tmp_path / "mill.mat").write_bytes(b"x")
    assert data.fetch(tmp_path) == tmp_path / "mill.mat"


def test_fetch_rejects_wrong_checksum(tmp_path):
    (tmp_path / "milling.zip").write_bytes(b"not the archive")
    with pytest.raises(RuntimeError):
        data.fetch(tmp_path)


# ---- real public data (skipped when the cache is empty) ----


def test_real_split_is_by_cutter(real_results):
    assert real_results["n_cutters"] == 15
    assert real_results["n_cuts"] == 166


def test_real_wear_beats_mean_predictor(real_results):
    assert real_results["wear_mae_mm"]["sensors+time"] < real_results["wear_mae_mean_baseline_mm"]


def test_real_wear_aware_matches_scrap_of_global_counter(real_results):
    s = real_results["policy"]["summary"]
    assert s["wear-aware"]["mean_cut_min"] > s["counter-global-min"]["mean_cut_min"]


def test_real_results_are_json_serialisable(real_results):
    assert json.loads(json.dumps(real_results))["limit_mm"] == 0.4


def test_real_demo_files_exist_in_repo():
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert (docs / "figure.png").exists()
