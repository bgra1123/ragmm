"""
Fit baselines and adapters, save held-out probabilities. Plan: PLAN.md (pre-registered).
Run from the repository root:  python research/sota2/fit_models.py SPLIT [--seeds 100]
"""
import argparse
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
for sub in ("polish", "linesearch", "model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parents[1]))
import loan_experiment as le  # noqa: E402
import linesearch_experiment as ls  # noqa: E402
import polish_experiments as pe  # noqa: E402

LAM, M_AVG = 0.01, 10


def head_fit(F, y):
    n = len(y)
    F = np.hstack([F, np.ones((n, 1))])

    def fg(w):
        m = y * (F @ w)
        return np.logaddexp(0, -m).mean() + 0.5 * LAM * w @ w, -(F.T @ (y * le.sigmoid(-m))) / n + LAM * w

    return minimize(fg, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-8, ftol=1e-15)).x


def onehot_leaves(Ltr, Lte):
    enc = [np.unique(Ltr[:, t]) for t in range(Ltr.shape[1])]
    f = lambda L: np.hstack([(L[:, t][:, None] == u[None, :]).astype(float) for t, u in enumerate(enc)])
    return f(Ltr), f(Lte)


class Family:
    name = ""
    refit = True

    def fit(self, X, y01, seed):
        raise NotImplementedError

    def proba(self, m, X):
        return m.predict_proba(X)[:, 1]

    def features(self, m, Xtr, Xte):
        raise NotImplementedError


class XGB(Family):
    name = "XGBoost"

    def fit(self, X, y01, seed):
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=seed, verbosity=0).fit(X, y01)

    def features(self, m, Xtr, Xte):
        return onehot_leaves(m.apply(Xtr), m.apply(Xte))


class LGB(Family):
    name = "LightGBM"

    def fit(self, X, y01, seed):
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, min_child_samples=10, n_jobs=1, random_state=seed, verbose=-1).fit(X, y01)

    def features(self, m, Xtr, Xte):
        return onehot_leaves(m.predict(Xtr, pred_leaf=True), m.predict(Xte, pred_leaf=True))


class RF(Family):
    name = "Random forest"

    def fit(self, X, y01, seed):
        return RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, n_jobs=1, random_state=seed).fit(X, y01)

    def features(self, m, Xtr, Xte):
        return onehot_leaves(m.apply(Xtr), m.apply(Xte))


class HGB(Family):
    name = "HistGradientBoosting"
    refit = False

    def fit(self, X, y01, seed):
        return HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.1, early_stopping=False, random_state=seed).fit(X, y01)


class MLP(Family):
    name = "MLP"

    def fit(self, X, y01, seed):
        return MLPClassifier(hidden_layer_sizes=(64, 64), solver="adam", learning_rate_init=1e-3, batch_size=32, max_iter=60,
                             alpha=1e-3, random_state=seed, tol=0, n_iter_no_change=1000).fit(X, y01)

    def features(self, m, Xtr, Xte):
        def acts(X):
            h = X
            for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
                h = np.maximum(h @ W + b, 0)
            return h
        return acts(Xtr), acts(Xte)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(oof_p, y01):
    z, t = logit(oof_p), 2.0 * y01 - 1.0
    w = np.zeros(2)
    A = np.stack([z, np.ones_like(z)], 1)
    for _ in range(50):
        s = le.sigmoid(t * (A @ w))
        g = -(A.T @ (t * (1 - s))) / len(z)
        H = (A.T * (s * (1 - s))) @ A / len(z) + 1e-6 * np.eye(2)
        w = w - np.linalg.solve(H, g)
    return w


def avg(P, m=M_AVG):
    return P.reshape(len(P) // m, m, -1).mean(1)


def run_family(fam, Xf, y, Xtef, seeds):
    y01 = (y > 0).astype(int)
    out = {"base": [], "refit": [], "platt": []}
    folds = np.arange(len(y)) % 3
    for s in range(seeds):
        m = fam.fit(Xf, y01, s)
        pt = fam.proba(m, Xtef)
        out["base"].append(pt)
        if fam.refit:
            Ftr, Fte = fam.features(m, Xf, Xtef)
            w = head_fit(Ftr, y)
            out["refit"].append(le.sigmoid(np.hstack([Fte, np.ones((len(Fte), 1))]) @ w))
        oof = np.zeros(len(y))
        perm = np.random.default_rng(s).permutation(len(y)) % 3
        for k in range(3):
            mk = fam.fit(Xf[perm != k], y01[perm != k], s + 1000 * (k + 1))
            oof[perm == k] = fam.proba(mk, Xf[perm == k])
        a, b = platt(oof, y01)
        out["platt"].append(le.sigmoid(a * logit(pt) + b))
    return {k: np.stack(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=int)
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    Xtr, ytr, Xte, yte = le.load_data(split_seed=args.split, n_test=300)
    Xf, Xtef = Xtr[:, :-1], Xte[:, :-1]
    n, d = Xtr.shape
    L, mu = le.constants(Xtr, LAM)
    wstar, _ = le.solve_exact(Xtr, ytr, LAM)
    arrays = {"y_test": yte, "logistic exact|base": le.sigmoid(Xte @ wstar)[None].repeat(2, 0)}
    # logistic baseline and its adapters
    idx1 = le.batch_indices(range(args.seeds), 2000, n, 16)
    W1 = le.run_gmm(Xtr, ytr, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0]
    arrays["logistic SGD|base"] = le.sigmoid(W1 @ Xte.T)
    K2 = round(5 * n / 16)
    sel, _ = pe.select_polish(Xtr, ytr, W1, K2, np.random.default_rng(0))
    idx2 = le.batch_indices(range(100_000, 100_000 + args.seeds), K2, n, 16)
    arrays["logistic SGD|polish adapter"] = le.sigmoid(pe.run_from(W1, Xtr, ytr, idx2, **sel) @ Xte.T)
    arrays["logistic SGD|line search"] = le.sigmoid(ls.gd(W1, Xtr, ytr, 5, "LS", 0, L)[0] @ Xte.T)
    print("logistic done", f"{time.time() - t0:.0f}s", flush=True)
    for fam in (XGB(), LGB(), RF(), HGB(), MLP()):
        P = run_family(fam, Xf, ytr, Xtef, args.seeds)
        for k, v in P.items():
            arrays[f"{fam.name}|{k}"] = v
        print(fam.name, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(HERE / "results" / f"probs_split{args.split}.npz", **arrays)


if __name__ == "__main__":
    main()
