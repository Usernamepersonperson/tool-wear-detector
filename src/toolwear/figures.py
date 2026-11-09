"""Result figures (dark palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = (
    "#38BDF8",
    "#34D399",
    "#FBBF24",
    "#F87171",
    "#64748B",
    "#A78BFA",
)
SOURCE = "Source: NASA Prognostics Data Repository, milling data (UC Berkeley BEST Lab, 2007)"


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def _style(ax, size: float) -> None:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _timeline(ax, ex: dict, size: float, lw: float = 2.4) -> None:
    t = np.array(ex["t"], dtype=float)
    vb = np.array([np.nan if v is None else v for v in ex["vb"]], dtype=float)
    ax.plot(t, ex["smooth"], color=ACCENT, lw=lw, label="estimated wear (from sensors)")
    ax.scatter(t, vb, color=TEXT, s=26, zorder=3, label="measured wear")
    ax.axhline(0.40, color=AMBER, lw=1.4, ls="--")
    ax.text(t[0], 0.415, "end-of-life 0.40 mm", color=AMBER, fontsize=size, va="bottom")
    ymax = max(0.8, float(np.nanmax(vb)) * 1.1)
    marks = [
        (ex["wear_aware_change_min"], GREEN, "wear-aware change"),
        (ex["counter_recipe_0.8x"], VIOLET, "fixed counter (0.8x)"),
    ]
    for x, col, lab in marks:
        ax.axvline(x, color=col, lw=2.2)
        ax.text(x, ymax * 0.97, " " + lab, color=col, fontsize=size, va="top", rotation=90)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, max(t.max(), ex["counter_recipe_0.8x"]) * 1.03)
    ax.set_xlabel("cutting time (min)", color=MUTED, fontsize=size)
    ax.set_ylabel("flank wear VB (mm)", color=MUTED, fontsize=size)
    _style(ax, size)


def _bars(ax, res: dict, size: float, short: bool = False) -> None:
    s = res["policy"]["summary"]
    keys = ["counter-global-min", "counter-recipe-0.6x", "counter-recipe-0.8x", "wear-aware"]
    labels = ["global\nmin counter", "recipe\n0.6x", "recipe\n0.8x", "wear-\naware"]
    cols = [GREY, VIOLET, VIOLET, GREEN]
    vals = [s[k]["mean_cut_min"] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=cols, width=0.62)
    for b, k in zip(bars, keys, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(vals) * 0.02,
            f"{b.get_height():.0f} min\nscrap risk {s[k]['scrap_risk'] * 100:.0f}%",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=size,
        )
    ax.set_xticks(range(len(keys)), labels)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("cutting per tool (min)", color=MUTED, fontsize=size)
    _style(ax, size)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", colors=TEXT)


def _scatter(ax, res: dict, size: float) -> None:
    pr = res["per_run"]
    vb = np.array([np.nan if v is None else v for v in pr["vb"]], dtype=float)
    pred = np.array(pr["pred"], dtype=float)
    m = np.isfinite(vb)
    ax.scatter(vb[m], pred[m], s=26, color=ACCENT, alpha=0.85)
    lim = max(float(vb[m].max()), float(pred[m].max())) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, lw=1.5)
    ax.axvline(0.40, color=AMBER, lw=1.2, ls="--")
    ax.axhline(0.40, color=AMBER, lw=1.2, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("measured wear VB (mm)", color=MUTED, fontsize=size)
    ax.set_ylabel("estimated wear (mm)", color=MUTED, fontsize=size)
    ax.text(
        0.04 * lim,
        0.93 * lim,
        f"MAE {res['wear_mae_mm']['sensors+time']:.3f} mm\n"
        f"(mean-only {res['wear_mae_mean_baseline_mm']:.3f})",
        color=TEXT,
        fontsize=size,
        va="top",
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 3},
        zorder=5,
    )
    _style(ax, size)


def _header(fig, title: str, sub: str, tsize: float, ssize: float) -> None:
    fig.text(0.03, 0.965, title, color=TEXT, fontsize=tsize, fontweight="bold", va="top")
    fig.text(0.03, 0.915, sub, color=MUTED, fontsize=ssize, va="top")


def make_figure(res: dict, path: Path) -> Path:
    """Main result figure, 1600x900 px."""
    plt.rcParams["font.family"] = _font()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    _header(
        fig,
        "Tool wear from spindle load, vibration and acoustic emission - tested on unseen cutters",
        SOURCE + ". Leave-one-cutter-out, 15 cutters.",
        19,
        13,
    )
    a = fig.add_axes([0.06, 0.12, 0.27, 0.68])
    b = fig.add_axes([0.40, 0.12, 0.29, 0.68])
    c = fig.add_axes([0.77, 0.12, 0.21, 0.68])
    _scatter(a, res, 12)
    ex = res["example"]
    _timeline(b, ex, 12)
    _bars(c, res, 11)
    a.set_title("Wear estimate vs measured", color=TEXT, fontsize=14, loc="left")
    b.set_title(f"Held-out cutter {ex['case']}: when to change", color=TEXT, fontsize=14, loc="left")
    c.set_title("Cutting per tool, scrap risk", color=TEXT, fontsize=14, loc="left")
    b.legend(loc="lower right", fontsize=11, facecolor=BG,
             edgecolor=GRID, labelcolor=TEXT)
    fig.text(
        0.03,
        0.02,
        "Scrap risk = share of tools changed after VB passed 0.40 mm. Counters set from the other "
        "cutters. Thresholds tuned on this data set. 'Parts' assume 3 min of cutting each.",
        color=MUTED,
        fontsize=11,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path


def make_post_figure(res: dict, path: Path) -> Path:
    """Phone-sized figure, exactly 960x700 px, two panels."""
    plt.rcParams["font.family"] = _font()
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    _header(
        fig,
        "Tool change: sensors vs a fixed counter",
        "Real open data: NASA milling wear, 15 cutters, each tested unseen",
        14,
        11,
    )
    ex = res["example"]
    a = fig.add_axes([0.095, 0.36, 0.87, 0.47])
    b = fig.add_axes([0.30, 0.075, 0.66, 0.17])
    _timeline(a, ex, 11, lw=3.0)
    a.legend(loc="lower right", fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    a.set_title(f"Held-out cutter {ex['case']}", color=TEXT, fontsize=12, loc="left")
    s = res["policy"]["summary"]
    keys = ["counter-global-min", "counter-recipe-0.6x", "wear-aware"]
    labels = ["global min counter", "recipe counter 0.6x", "wear-aware"]
    cols = [GREY, VIOLET, GREEN]
    vals = [s[k]["mean_cut_min"] for k in keys]
    b.barh(range(3), vals, color=cols, height=0.62)
    b.set_yticks(range(3), labels)
    b.invert_yaxis()
    for i, k in enumerate(keys):
        b.text(vals[i] + 0.6, i, f"{vals[i]:.0f} min, scrap {s[k]['scrap_risk'] * 100:.0f}%",
               color=TEXT, fontsize=11, va="center")
    b.set_xlim(0, max(vals) * 1.55)
    _style(b, 11)
    b.tick_params(axis="y", colors=TEXT)
    b.grid(axis="y", visible=False)
    b.set_xlabel("mean cutting per tool (min)", color=MUTED, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path
