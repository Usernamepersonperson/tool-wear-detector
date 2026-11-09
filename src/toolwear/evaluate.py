"""Leave-one-cutter-out evaluation: every test cutter is unseen during training."""

from __future__ import annotations

import numpy as np

from toolwear import alarm, features, policy, rul
from toolwear.data import Run
from toolwear.model import VARIANTS, Table, WearModel

LIMIT = 0.40  # mm flank wear: end-of-life criterion used here
SEVERE = 0.60  # mm: severe wear, chipping / breakage risk zone (proxy label)
FPR = 0.05  # fixed false-alarm rate for the alarms
MARGIN = 0.15  # change the tool when smoothed wear reaches LIMIT - MARGIN
PART_MIN = 3.0  # assumed minutes of cutting per part (median log interval is 3 min)
MARGIN_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
LATE_RUL_MIN = 15.0


def loco_predict(table: Table, variant: str, seed: int = 0) -> np.ndarray:
    """Out-of-fold wear predictions: each cutter predicted by a model trained on the others."""
    pred = np.full(len(table), np.nan)
    for c in table.cutters():
        te = np.where(table.case == c)[0]
        tr = np.where(table.case != c)[0]
        pred[te] = WearModel(variant, seed=seed).fit(table, tr).predict(table, te)
    return pred


def loco_mean_baseline(table: Table) -> np.ndarray:
    pred = np.full(len(table), np.nan)
    for c in table.cutters():
        te = table.case == c
        tr = (~te) & np.isfinite(table.vb)
        pred[te] = float(np.mean(table.vb[tr]))
    return pred


def mae(pred: np.ndarray, vb: np.ndarray) -> float:
    m = np.isfinite(vb) & np.isfinite(pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(pred[m] - vb[m])))


def _tool_series(table: Table, pred: np.ndarray) -> dict[int, dict]:
    out = {}
    for c in table.cutters():
        r = table.rows(c)
        out[c] = {
            "recipe": (float(table.process[r[0], 2]), float(table.process[r[0], 0]),
                       float(table.process[r[0], 1])),
            "rows": r,
            "t": table.time[r],
            "vb": table.vb[r],
            "pred": pred[r],
            "smooth": rul.smooth_wear(pred[r]),
            "life": rul.time_to_limit(table.time[r], table.vb[r], LIMIT),
            "severe_t": rul.time_to_limit(table.time[r], table.vb[r], SEVERE),
        }
    return out


def _summ(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def evaluate_rul(series: dict[int, dict]) -> dict:
    lives = {c: s["life"] for c, s in series.items() if s["life"] is not None}
    ours, base_mean, late_ours, late_base = [], [], [], []
    for c, s in series.items():
        if s["life"] is None:
            continue
        train = [v for k, v in lives.items() if k != c]
        mean_life = policy.counter_life(train, "mean")
        for j in range(len(s["t"])):
            true = s["life"] - s["t"][j]
            if true <= 0:
                continue
            est = rul.estimate_rul(s["t"][: j + 1], s["smooth"][: j + 1], LIMIT)
            if not np.isfinite(est):
                continue
            e1, e2 = abs(est - true), abs(max(mean_life - s["t"][j], 0.0) - true)
            ours.append(e1)
            base_mean.append(e2)
            if true <= LATE_RUL_MIN:
                late_ours.append(e1)
                late_base.append(e2)
    return {
        "n_points": len(ours),
        "rul_mae_min": _summ(ours),
        "counter_rul_mae_min": _summ(base_mean),
        "late_n_points": len(late_ours),
        "late_rul_mae_min": _summ(late_ours),
        "late_counter_rul_mae_min": _summ(late_base),
    }


def evaluate_alarm(table: Table, series: dict[int, dict]) -> dict:
    neg, pos = [], []
    for s in series.values():
        for v, sc in zip(s["vb"], s["smooth"], strict=True):
            if not np.isfinite(v):
                continue
            if v < LIMIT:
                neg.append(sc)
            elif v >= SEVERE:
                pos.append(sc)
    thr = alarm.threshold_at_fpr(np.array(neg), FPR)
    leads, caught, reach = [], 0, 0
    for s in series.values():
        if s["severe_t"] is None:
            continue
        reach += 1
        first_severe_run = next(t for t, v in zip(s["t"], s["vb"], strict=True) if v >= SEVERE)
        a = alarm.first_alarm(s["t"], s["smooth"], thr)
        if a is not None and a <= first_severe_run:
            caught += 1
            leads.append(s["severe_t"] - a)
    return {
        "threshold_mm": thr,
        "fpr_target": FPR,
        "n_neg_runs": len(neg),
        "n_pos_runs": len(pos),
        "false_alarm_rate": alarm.false_alarm_rate(np.array(neg), thr),
        "run_recall": alarm.recall_at(np.array(pos), thr),
        "tools_reaching_severe": reach,
        "tools_caught": caught,
        "tool_recall": caught / reach if reach else float("nan"),
        "median_lead_min": float(np.median(leads)) if leads else float("nan"),
        "mean_lead_min": _summ(leads),
    }


def evaluate_policy(series: dict[int, dict], margin: float = MARGIN) -> dict:
    lives = {c: s["life"] for c, s in series.items() if s["life"] is not None}
    names = (
        "counter-global-min",
        "counter-recipe-0.6x",
        "counter-recipe-0.8x",
        "counter-recipe-1.0x",
        "wear-aware",
    )
    rows: dict[str, list[dict]] = {n: [] for n in names}
    censored = 0
    for c, s in series.items():
        if s["life"] is None:
            continue
        train = {k: v for k, v in lives.items() if k != c}
        known = [(series[k]["recipe"], v) for k, v in train.items()]
        fixed = {
            "counter-global-min": policy.counter_life(list(train.values()), "conservative"),
            "counter-recipe-0.6x": policy.recipe_counter(known, s["recipe"], 0.6),
            "counter-recipe-0.8x": policy.recipe_counter(known, s["recipe"], 0.8),
            "counter-recipe-1.0x": policy.recipe_counter(known, s["recipe"], 1.0),
        }
        for name, life in fixed.items():
            rows[name].append(policy.outcome(life, s["life"], PART_MIN))
        t_change, seen = policy.change_time(s["t"], s["smooth"], LIMIT - margin)
        censored += 0 if seen else 1
        rows["wear-aware"].append(policy.outcome(t_change, s["life"], PART_MIN))
    summary = {}
    for name, rs in rows.items():
        summary[name] = {
            "mean_cut_min": _summ([r["cut_min"] for r in rs]),
            "mean_parts": _summ([r["parts"] for r in rs]),
            "scrap_risk": _summ([r["late"] for r in rs]),
            "mean_overrun_min": _summ([r["overrun_min"] for r in rs]),
            "mean_unused_life_min": _summ([r["unused_min"] for r in rs]),
        }
    wa = summary["wear-aware"]
    vs = {}
    for name in names[:-1]:
        base = summary[name]
        vs[name] = {
            "extra_parts_per_tool": wa["mean_parts"] - base["mean_parts"],
            "life_gain_pct": 100.0 * (wa["mean_cut_min"] / base["mean_cut_min"] - 1.0),
            "scrap_risk_delta": wa["scrap_risk"] - base["scrap_risk"],
        }
    return {
        "n_tools": len(rows["wear-aware"]),
        "margin_mm": margin,
        "part_min": PART_MIN,
        "wear_aware_unobserved_trigger": censored,
        "summary": summary,
        "wear_aware_vs": vs,
    }


def example_tool(series: dict[int, dict], alarm_thr: float, margin: float = MARGIN) -> dict:
    """Timeline of the held-out cutter with the most cuts, for the figures."""
    lives = {c: s["life"] for c, s in series.items() if s["life"] is not None}
    c = max(lives, key=lambda k: len(series[k]["t"]))
    s = series[c]
    train = {k: v for k, v in lives.items() if k != c}
    known = [(series[k]["recipe"], v) for k, v in train.items()]
    t_change, _ = policy.change_time(s["t"], s["smooth"], LIMIT - margin)
    return {
        "case": int(c),
        "t": s["t"].tolist(),
        "vb": [None if not np.isfinite(v) else float(v) for v in s["vb"]],
        "smooth": s["smooth"].tolist(),
        "true_life_min": s["life"],
        "alarm_at_min": alarm.first_alarm(s["t"], s["smooth"], alarm_thr),
        "wear_aware_change_min": t_change,
        "counter_global_min": policy.counter_life(list(train.values()), "conservative"),
        "counter_recipe_0.8x": policy.recipe_counter(known, s["recipe"], 0.8),
    }


IMPULSE_LEVELS = {"mild": (6.0, 0.1), "moderate": (12.0, 0.2), "severe": (24.0, 0.4)}


def evaluate_impulse(table: Table, runs: list[Run], seed: int = 0) -> dict:
    """Alarm test with synthetic breakage injected into real held-out cuts (labelled injected).

    The threshold is set at a fixed false-alarm rate on the clean cuts; recall is reported for
    three injected event sizes (impulse burst in sigma of the channel, load step in share of range).
    """
    rng = np.random.default_rng(seed)
    sig = {(r.case, r.run): r.signals for r in runs}
    clean: list[float] = []
    inj: dict[str, list[float]] = {k: [] for k in IMPULSE_LEVELS}
    for c in table.cutters():
        r = table.rows(c)
        for j in range(3, len(r)):
            hist = table.feats[r[max(0, j - 5) : j]]
            clean.append(alarm.impulse_score(hist, table.feats[r[j]]))
            base = sig[(int(table.case[r[j]]), int(table.run[r[j]]))]
            for name, (burst, step) in IMPULSE_LEVELS.items():
                s2 = alarm.inject_breakage(base, rng, burst, step)
                inj[name].append(alarm.impulse_score(hist, features.run_features(s2)))
    clean_a = np.array(clean)
    thr = alarm.threshold_at_fpr(clean_a, FPR)
    return {
        "n_clean": int(clean_a.size),
        "threshold_z": thr,
        "false_alarm_rate": alarm.false_alarm_rate(clean_a, thr),
        "recall": {k: alarm.recall_at(np.array(v), thr) for k, v in inj.items()},
        "levels": {k: {"burst_sigma": b, "load_step": st} for k, (b, st) in IMPULSE_LEVELS.items()},
        "label": "injected synthetic breakage on real cuts",
    }


def run_all(table: Table, runs: list[Run] | None = None, seed: int = 0) -> dict:
    preds = {v: loco_predict(table, v, seed) for v in VARIANTS}
    main = preds["sensors+time"]
    series = _tool_series(table, main)
    res = {
        "n_cuts": len(table),
        "n_cutters": len(table.cutters()),
        "n_labelled": int(np.isfinite(table.vb).sum()),
        "limit_mm": LIMIT,
        "severe_mm": SEVERE,
        "wear_mae_mm": {v: mae(p, table.vb) for v, p in preds.items()},
        "wear_mae_mean_baseline_mm": mae(loco_mean_baseline(table), table.vb),
        "rul": evaluate_rul(series),
        "alarm": evaluate_alarm(table, series),
        "policy": evaluate_policy(series),
        "policy_margin_sweep": {
            f"{m:.2f}": evaluate_policy(series, m)["summary"]["wear-aware"] for m in MARGIN_GRID
        },
        "per_run": {
            "case": table.case.tolist(),
            "time": table.time.tolist(),
            "vb": [None if not np.isfinite(v) else float(v) for v in table.vb],
            "pred": main.tolist(),
            "smooth": np.concatenate([series[c]["smooth"] for c in table.cutters()]).tolist(),
            "order": np.concatenate([series[c]["rows"] for c in table.cutters()]).tolist(),
        },
        "tool_lives_min": {
            str(c): s["life"] for c, s in series.items() if s["life"] is not None
        },
    }
    res["example"] = example_tool(series, res["alarm"]["threshold_mm"])
    if runs is not None:
        res["impulse"] = evaluate_impulse(table, runs, seed)
    return res
