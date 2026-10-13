"""
Command line:  ragmm DATA.csv --target COLUMN [--positive VALUE] [--method ragmm|sgd|careful]
                     [--drop COL ...] [--report report.md] [--retrains 200] [--model model.json]

Trains on all rows with the chosen method and writes the model (coefficients, feature names, parameters). With
--report it also retrains many times on a held-out split and writes the tail report (Markdown, plus JSON next to it).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from .classifier import METHODS, RiskAverseClassifier
from .tabular import TabularEncoder, binary_target
from .tail_report import DISCLAIMER, tail_report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ragmm", description="Train a risk-averse logistic regression on a CSV.")
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--positive", default=None, help="target value that counts as positive")
    ap.add_argument("--drop", nargs="*", default=[], help="columns to leave out (e.g. ids, protected attributes)")
    ap.add_argument("--method", choices=METHODS, default="ragmm")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--report", default=None, help="write a tail report to this Markdown file")
    ap.add_argument("--retrains", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        ap.error(f"target column {args.target!r} not found; columns: {list(df.columns)}")
    y, pos = binary_target(df[args.target], args.positive)
    enc = TabularEncoder(drop=args.drop).fit(df.drop(columns=[args.target]))
    X = enc.transform(df.drop(columns=[args.target]))
    clf = RiskAverseClassifier(method=args.method, lam=args.lam, steps=args.steps, batch_size=args.batch).fit(X, y)
    model = dict(method=args.method, target=args.target, positive=str(pos), params=clf.params_,
                 info={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in clf.info_.items()},
                 features=enc.feature_names_, coefficients=[float(c) for c in clf.coef_],
                 training_accuracy=float((clf.predict(X) == y).mean()))
    pathlib.Path(args.model).write_text(json.dumps(model, indent=2))
    print(f"trained {args.method} on {len(y)} rows, {X.shape[1]} features; "
          f"alpha={clf.params_['alpha']:.4g} beta={clf.params_['beta']:.3g} gamma={clf.params_['gamma']:.3g}; "
          f"training accuracy {model['training_accuracy']:.3f}; wrote {args.model}")
    if args.report:
        rep = tail_report(df, args.target, positive=pos, drop=args.drop, retrains=args.retrains,
                          threshold=args.threshold, lam=args.lam, steps=args.steps, batch_size=args.batch)
        path = pathlib.Path(args.report)
        path.write_text(rep.to_markdown())
        path.with_suffix(".json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
        print(f"wrote {path} and {path.with_suffix('.json')}")
    print(DISCLAIMER, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
