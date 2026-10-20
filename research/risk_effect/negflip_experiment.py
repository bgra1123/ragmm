"""Negative-flip test (PLAN.md section 10). Run from the repository root:  python research/risk_effect/negflip_experiment.py german compas pima heart
Output: results/negflip_<dataset>.json"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepmatch_experiment as sm  # noqa: E402
import risk_effect_experiment as re_  # noqa: E402
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

N_BOOT = 1000
sig = le.sigmoid


def flips(W, D1, good, groups):
    D = sig(W @ sm_Xte.T) >= 0.5
    rb, ra = D1 == good[None], D == good[None]
    out = dict(neg=(rb & ~ra).mean(1), pos=(~rb & ra).mean(1), churn=(D != D1).mean(1))
    if groups:
        out["gneg"] = np.stack([(rb[:, mk] & ~ra[:, mk]).mean(1) for mk in groups.values()], 1)
    return out


def summ(f):
    s = dict(neg=float(f["neg"].mean()), pos=float(f["pos"].mean()), net=float((f["pos"] - f["neg"]).mean()), churn=float(f["churn"].mean()))
    if "gneg" in f:
        s["worst_group_neg"] = float(f["gneg"].mean(0).max())
    return s


def boot(fa, fb, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    S = len(fa["neg"])
    acc = {k: [] for k in ("neg", "pos", "net", "churn", "worst_group_neg")}
    for _ in range(n):
        i = rng.integers(0, S, S)
        acc["neg"].append(fa["neg"][i].mean() - fb["neg"][i].mean())
        acc["pos"].append(fa["pos"][i].mean() - fb["pos"][i].mean())
        acc["net"].append((fa["pos"][i] - fa["neg"][i]).mean() - (fb["pos"][i] - fb["neg"][i]).mean())
        acc["churn"].append(fa["churn"][i].mean() - fb["churn"][i].mean())
        if "gneg" in fa:
            acc["worst_group_neg"].append(fa["gneg"][i].mean(0).max() - fb["gneg"][i].mean(0).max())
    return {k: [float(np.mean(v)), *[float(x) for x in np.quantile(v, [0.025, 0.975])]] for k, v in acc.items() if v}


def run(name):
    global sm_Xte
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    sm_Xte = Xte
    n, d = X.shape
    L, mu = le.constants(X, sm.LAM)
    good = yte > 0
    groups = None
    if name == "german":
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        raw = df[test_mask(yall, 0)].reset_index(drop=True)
        groups = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
        groups = {g: mk for g, mk in groups.items() if mk.sum() >= 30}
    out = dict(dataset=name, seeds=sm.SEEDS, n=n, stage1={})
    for opt in ("SGD", "Adam"):
        W1 = sm.stage1(opt, X, y, L)
        D1 = sig(W1 @ Xte.T) >= 0.5
        lr0 = 1 / L if opt == "SGD" else 0.01
        snaps_L, al_L = sm.trajectory(W1, X, y, 5, "LS")
        R_runs = {(r, th): sm.trajectory(W1, X, y, 5, r, th) for r in ("R1", "R3") for th in sm.THETAS}
        res = {}
        for B in sm.BUDGETS:
            F = {"L": flips(snaps_L[B], D1, good, groups)}
            idx2 = le.batch_indices(range(100_000, 100_000 + sm.SEEDS), max(1, round(B * n / sm.BATCH)), n, sm.BATCH)
            for nm, mode in (("C1", "const"), ("C2", "decay"), ("C3", "avg")):
                F[nm] = flips(re_.continue_training(W1, X, y, idx2, opt, lr0, mode), D1, good, groups)
            for (r, th), (snaps, al) in R_runs.items():
                k = f"{r}(theta={th:g})"
                F[k] = flips(snaps[B], D1, good, groups)
                c = float(al[:B].mean() / al_L[:B].mean())
                F[f"M1[{k}]"] = flips(sm.trajectory(W1, X, y, B, "LS", scale=c)[0][B], D1, good, groups)
                F[f"M2[{k}]"] = flips(sm.trajectory(W1, X, y, B, schedule=al[:B].mean(1))[0][B], D1, good, groups)
            best_c = min(("C1", "C2", "C3"), key=lambda c_: F[c_]["neg"].mean())
            tests = {}
            for k in [k for k in F if k.startswith("R")]:
                for ref in ("L", best_c, "C1", f"M1[{k}]", f"M2[{k}]"):
                    tests[f"{k} vs {ref.split('[')[0] if ref.startswith('M') else ref}"] = boot(F[k], F[ref])
            res[str(B)] = dict(best_continuation=best_c, arms={k: summ(v) for k, v in F.items()}, tests=tests)
            print(name, opt, B, f"[{time.time() - t0:.0f}s]", flush=True)
        out["stage1"][opt] = dict(stage1_error=float((D1 != good[None]).mean()), budgets=res)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"negflip_{name}.json").write_text(json.dumps(run(name), indent=2))
