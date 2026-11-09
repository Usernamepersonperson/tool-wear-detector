"""Plain-text report of the evaluation results."""

from __future__ import annotations


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def format_report(res: dict) -> str:
    w, r, a, p = res["wear_mae_mm"], res["rul"], res["alarm"], res["policy"]
    lines = [
        "toolwear demo: NASA milling-wear data (real, open), leave-one-cutter-out",
        f"{res['n_cutters']} cutters, {res['n_cuts']} cuts, {res['n_labelled']} with measured VB",
        "",
        "Wear estimate (mean absolute error, mm VB):",
        f"  sensors + time + recipe : {w['sensors+time']:.3f}",
        f"  time + recipe only      : {w['time-only']:.3f}",
        f"  sensors + recipe only   : {w['sensors-only']:.3f}",
        f"  predict the mean        : {res['wear_mae_mean_baseline_mm']:.3f}",
        "",
        f"Remaining useful life to VB {res['limit_mm']:.2f} mm (MAE, min; {r['n_points']} points):",
        f"  trend from sensors      : {r['rul_mae_min']:.1f}  "
        f"(last 15 min of life: {r['late_rul_mae_min']:.1f})",
        f"  mean-life counter       : {r['counter_rul_mae_min']:.1f}  "
        f"(last 15 min of life: {r['late_counter_rul_mae_min']:.1f})",
        "",
        f"Severe-wear alarm (VB >= {res['severe_mm']:.2f} mm is the positive class, proxy for chipping):",
        f"  threshold {a['threshold_mm']:.3f} mm, false-alarm rate {_pct(a['false_alarm_rate'])} "
        f"(target {_pct(a['fpr_target'])}), run recall {_pct(a['run_recall'])}",
        f"  tools caught {a['tools_caught']}/{a['tools_reaching_severe']}, "
        f"median lead {a['median_lead_min']:.1f} min",
    ]
    if "impulse" in res:
        i = res["impulse"]
        rec = ", ".join(f"{k} {_pct(v)}" for k, v in i["recall"].items())
        lines += [
            "",
            f"Impulse (breakage) alarm, {i['label']}:",
            f"  false-alarm rate {_pct(i['false_alarm_rate'])} on {i['n_clean']} clean cuts; "
            f"recall {rec}",
        ]
    lines += [
        "",
        f"Tool change policy ({p['n_tools']} tools, trigger = VB {res['limit_mm']:.2f} mm minus "
        f"{p['margin_mm']:.2f} mm, 1 part = {p['part_min']:.0f} min):",
        f"  {'policy':<22}{'cut min':>9}{'parts':>8}{'scrap':>8}{'unused':>9}",
    ]
    for name, s in p["summary"].items():
        lines.append(
            f"  {name:<22}{s['mean_cut_min']:>9.1f}{s['mean_parts']:>8.1f}"
            f"{_pct(s['scrap_risk']):>8}{s['mean_unused_life_min']:>9.1f}"
        )
    for name, v in p["wear_aware_vs"].items():
        lines.append(
            f"  wear-aware vs {name}: {v['extra_parts_per_tool']:+.1f} parts/tool "
            f"({v['life_gain_pct']:+.0f}% life), scrap risk {100 * v['scrap_risk_delta']:+.0f} pts"
        )
    return "\n".join(lines)
