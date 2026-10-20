"""
Risk-adjusted adapter vs one more epoch of the same training. Plan: PLAN.md (pre-registered, section 2).
Run from the repository root:  python research/risk_effect/risk_effect_experiment.py german compas pima heart
Output: results/<dataset>.json
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for sub in ("linesearch", "polish", "optimizers", "model_multiplicity", "gate0", "groups"):
    sys.path.insert(0, str(HERE.parent / sub))
import loan_experiment as le  # noqa: E402
import linesearch_experiment as ls  # noqa: E402
import optimizer_experiments as oe  # noqa: E402
import run_gate0 as g0  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

LAM, BATCH, K1, SEEDS = 0.01, 16, 2000, 200
BUDGETS = (1, 2, 5, 20)
N_BOOT = 1000
STAGE1 = ("SGD", "Adam")
R_ARMS = {"R risk first-order (theta=1)": ("R1", 1.0), "R risk exact-change (theta=5)": ("R3", 5.0)}
L_ARM = {"L exact line search": ("LS", 0.0)}


def stage1(opt, X, y, L, seeds):
    n = X.shape[0]
    idx = le.batch_indices(range(seeds), K1, n, BATCH)
    if opt == "SGD":
        return le.run_gmm(X, y, LAM, idx, alpha=1 / L, beta=0.0, gamma=0.0)[0]
    return oe.run_opt("Adam", X, y, idx, 0.01)


def continue_training(W0, X, y, idx, opt, lr0, mode):
    """Same optimizer for the extra budget. mode: 'const' | 'decay' (linear to 0) | 'avg' (iterate average).
    Adam restarts its moment estimates (fresh optimizer state), SGD has none."""
    S, K, b = idx.shape
    W = W0.copy()
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    avg = np.zeros_like(W)
    for k in range(K):
        Xb, yb = X[idx[:, k]], y[idx[:, k]]
        coef = -yb * le.sigmoid(-np.einsum("sbd,sd->sb", Xb, W) * yb)
        g = np.einsum("sbd,sb->sd", Xb, coef) / b + LAM * W
        lr = lr0 * (1 - k / K) if mode == "decay" else lr0
        if opt == "SGD":
            W = W - lr * g
        else:
            t = k + 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            W = W - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if mode == "avg":
            avg += (W - avg) / (k + 1)
    return avg if mode == "avg" else W


def disagreement(D):
    p = D.mean(0)
    return float(np.mean(2 * p * (1 - p)) * len(D) / (len(D) - 1))


def changed(D, ex):
    c = (D != ex[None]).sum(1)
    return float(c.mean()), float(c[c >= np.quantile(c, 0.95)].mean())


def arm_metrics(W, W1, X, y, Xte, yte, wstar, fstar, groups):
    base = ls.metrics(W, W1, X, y, Xte, wstar, fstar)
    D = le.sigmoid(W @ Xte.T) >= 0.5
    good = yte > 0
    base["wrong"] = float((D != good[None]).mean())
    if groups:
        gw = {g: float((D[:, mk] != good[mk][None]).mean()) for g, mk in groups.items()}
        base["group_wrong"] = gw
        base["worst_group_wrong"] = max(gw.values())
    return base, D


def boot(Da, Db, ex, stat, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    S = len(Da)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, S, S)
        diffs.append(stat(Da[i], ex) - stat(Db[i], ex))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(stat(Da, ex) - stat(Db, ex)), float(lo), float(hi)


def run(name, seeds=SEEDS):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    ex = le.sigmoid(Xte @ wstar) >= 0.5
    groups = None
    if name == "german":
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        raw = df[test_mask(yall, 0)].reset_index(drop=True)
        assert np.array_equal(np.where(raw[le.TARGET] == "good", 1.0, -1.0), yte)
        groups = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
        groups = {g: mk for g, mk in groups.items() if mk.sum() >= 30}
    out = dict(dataset=name, seeds=seeds, n=n, d=d, L=L, stage1={}, deviation_notes=[])
    for opt in STAGE1:
        W1 = stage1(opt, X, y, L, seeds)
        lr0 = 1 / L if opt == "SGD" else 0.01
        m1, D1 = arm_metrics(W1, W1, X, y, Xte, yte, wstar, fstar, groups)
        res = dict(stage1=dict(metrics=m1, disagreement=disagreement(D1), changed=changed(D1, ex)), budgets={})
        for B in BUDGETS:
            K2 = max(1, round(B * n / BATCH))
            idx2 = le.batch_indices(range(100_000, 100_000 + seeds), K2, n, BATCH)
            arms, decs, secs = {}, {}, {}
            for nm, mode in (("C1 continue", "const"), ("C2 continue, decayed", "decay"), ("C3 continue, averaged", "avg")):
                t = time.time()
                W = continue_training(W1, X, y, idx2, opt, lr0, mode)
                secs[nm] = time.time() - t
                arms[nm], decs[nm] = arm_metrics(W, W1, X, y, Xte, yte, wstar, fstar, groups)
            for nm, (rule, th) in {**R_ARMS, **L_ARM}.items():
                t = time.time()
                W, al = ls.gd(W1, X, y, B, rule, th, L)
                secs[nm] = time.time() - t
                arms[nm], decs[nm] = arm_metrics(W, W1, X, y, Xte, yte, wstar, fstar, groups)
                arms[nm]["mean_alpha_times_L"] = float(al.mean() * L)
            for nm in arms:
                arms[nm]["disagreement"] = disagreement(decs[nm])
                arms[nm]["changed_mean"], arms[nm]["changed_worst5"] = changed(decs[nm], ex)
                arms[nm]["seconds_all_seeds"] = secs[nm]
            cont = ["C1 continue", "C2 continue, decayed", "C3 continue, averaged"]
            best = min(cont, key=lambda c: arms[c]["disagreement"])
            tests = {}
            for r in R_ARMS:
                for ref in (best, "C1 continue", "L exact line search"):
                    tests[f"{r} vs {ref}"] = dict(
                        disagreement=boot(decs[r], decs[ref], ex, lambda D, e: disagreement(D)),
                        changed_mean=boot(decs[r], decs[ref], ex, lambda D, e: changed(D, e)[0]),
                        wrong=boot(decs[r], decs[ref], ex, lambda D, e: float((D != (yte > 0)[None]).mean())))
            res["budgets"][str(B)] = dict(K2=K2, best_continuation=best, arms=arms, tests=tests)
            print(name, opt, f"B={B}", {k[:14]: round(v["disagreement"], 4) for k, v in arms.items()},
                  f"[{time.time() - t0:.0f}s]", flush=True)
        out["stage1"][opt] = res
    return out


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"{name}.json").write_text(json.dumps(run(name), indent=2))
