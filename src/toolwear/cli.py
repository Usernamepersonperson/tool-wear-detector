"""Command line: python -m toolwear demo | fetch | figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolwear import data, evaluate, figures, model, report


def _run(cache: Path) -> dict:
    mat = data.fetch(cache)
    runs = data.load_runs(mat)
    table = model.build_table(runs)
    return evaluate.run_all(table, runs)


def cmd_demo(args: argparse.Namespace) -> int:
    res = _run(Path(args.cache))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(report.format_report(res))
    if not args.no_figures:
        figures.make_figure(res, out / "figure.png")
        figures.make_post_figure(res, out / "post-figure.png")
        print(f"\nwrote {out / 'results.json'}, {out / 'figure.png'}, {out / 'post-figure.png'}")
    print("\nData: " + data.CITATION)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    print(data.fetch(Path(args.cache)))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    figures.make_figure(res, out / "figure.png")
    figures.make_post_figure(res, out / "post-figure.png")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toolwear", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="fetch data, evaluate, write results and figures")
    d.add_argument("--cache", default="cache")
    d.add_argument("--out", default="docs")
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(func=cmd_demo)
    f = sub.add_parser("fetch", help="download the public data set into the cache folder")
    f.add_argument("--cache", default="cache")
    f.set_defaults(func=cmd_fetch)
    g = sub.add_parser("figures", help="redraw figures from docs/results.json")
    g.add_argument("--results", default="docs/results.json")
    g.add_argument("--out", default="docs")
    g.set_defaults(func=cmd_figures)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
