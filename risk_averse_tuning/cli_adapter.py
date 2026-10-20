"""
Command line:  ragmm-adapter DATA.csv --target COLUMN [--positive VALUE] [--model sgd|logistic|xgboost|lightgbm|
                             random_forest|mlp] [--group-columns COL ...] [--seeds 20] [--report stability.md]

Splits the data (70/30, stratified), retrains the chosen model with --seeds different random seeds, once as is and once
wrapped in the StabilityAdapter, and writes a before/after stability report (Markdown, plus JSON next to it).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

from .adapter import DISCLAIMER, StabilityAdapter, compare_stability, stability_check
from .tabular import TabularEncoder, binary_target, stratified_split

MODELS = ("sgd", "logistic", "xgboost", "lightgbm", "random_forest", "mlp")


def make_factory(name, lam=0.01):
    """seed -> unfitted model, for the built-in model choices."""
    if name == "sgd":
        from sklearn.linear_model import SGDClassifier
        return lambda s: SGDClassifier(loss="log_loss", alpha=lam, max_iter=5, tol=None, random_state=s)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return lambda s: LogisticRegression(C=1.0, max_iter=1000, random_state=s)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return lambda s: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                                       colsample_bytree=0.8, random_state=s, verbosity=0, n_jobs=1)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return lambda s: LGBMClassifier(n_estimators=200, num_leaves=15, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=s, verbose=-1, n_jobs=1)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return lambda s: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=s, n_jobs=1)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return lambda s: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=300, random_state=s)
    raise ValueError(f"model must be one of {MODELS}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ragmm-adapter", description="Before/after stability report for the adapter.")
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--positive", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--model", choices=MODELS, default="xgboost")
    ap.add_argument("--group-columns", nargs="*", default=[], help="report each value of these columns as a group")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=10, help="copies averaged by the adapter (tree models, MLP)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test-size", type=float, default=0.3, help="share of rows held out")
    ap.add_argument("--report", default="stability.md")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        ap.error(f"target column {args.target!r} not found; columns: {list(df.columns)}")
    y, pos = binary_target(df[args.target], args.positive)
    y01 = (y > 0).astype(int)
    feats = df.drop(columns=[args.target])
    te = stratified_split(y, args.test_size, seed=0)
    tr = ~te
    enc = TabularEncoder(drop=args.drop).fit(feats[tr])
    Xtr, Xte = enc.transform(feats[tr])[:, :-1], enc.transform(feats[te])[:, :-1]
    groups = {}
    for c in args.group_columns:
        for v in feats[c][te].dropna().unique():
            groups[f"{c} = {v}"] = (feats[c][te] == v).to_numpy()
    base = make_factory(args.model)
    adapted = lambda s: StabilityAdapter(base(s), n_seeds=args.n_seeds, calibrate=args.calibrate, random_state=s)  # noqa: E731
    kw = dict(seeds=args.seeds, groups=groups, threshold=args.threshold)
    before = stability_check(base, Xtr, y01[tr], Xte, y01[te], name=f"{args.model}", **kw)
    after = stability_check(adapted, Xtr, y01[tr], Xte, y01[te], name=f"{args.model} + adapter", **kw)
    md = [f"# Stability report: {args.model} on {pathlib.Path(args.data).name}", "",
          f"{int(tr.sum())} training rows, {int(te.sum())} held-out rows, {args.seeds} retrains each, decision threshold "
          f"{args.threshold}. Positive class: {pos}.", "", compare_stability([before, after]), ""]
    for g in groups:
        if g in before.groups:
            md += [f"### {g}", "", compare_stability([before, after], group=g), ""]
    md += ["## Largest effects of training randomness (before the adapter)", ""] + [f"- {s}" for s in before.statements()]
    md += ["", DISCLAIMER]
    path = pathlib.Path(args.report)
    path.write_text("\n".join(md))
    path.with_suffix(".json").write_text(json.dumps([before.to_dict(), after.to_dict()], indent=2))
    print("\n".join(md[:6]))
    print(f"wrote {path} and {path.with_suffix('.json')}")
    print(DISCLAIMER, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
