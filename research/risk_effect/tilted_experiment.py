"""
Experiment B (PLAN.md section 4): does a tilted (entropic-risk) objective change the final model beyond reweighting?
Run from the repository root:  python research/risk_effect/tilted_experiment.py german compas pima heart
Output: results/tilted_<dataset>.json

Tilted loss  F_theta(w) = (1/theta) log sum_i s_i exp(theta l_i(w)) + (lambda/2)||w||^2   (s = row weights summing to 1;
theta = 0 is the weighted mean). Convex and lambda-strongly convex; solved exactly by Newton, so there is no seed noise.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for sub in ("model_multiplicity", "gate0", "groups"):
    sys.path.insert(0, str(HERE.parent / sub))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

LAM = 0.01
THETAS = (0.5, 1.0, 2.0, 5.0)
SPLITS = 20
N_BOOT = 1000
MIN_POOLED_GROUP_ROWS = 200


def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def tilted_solve(X, y, s, theta, lam=LAM, tol=1e-9, max_iter=60, w0=None):
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy()
    s = s / s.sum()
    ls = np.log(np.maximum(s, 1e-300))

    def F(w):
        l = np.logaddexp(0.0, -y * (X @ w))
        if theta == 0:
            return float(s @ l + 0.5 * lam * w @ w)
        z = theta * l + ls
        m = z.max()
        return float((m + np.log(np.exp(z - m).sum())) / theta + 0.5 * lam * w @ w)

    for _ in range(max_iter):
        m = y * (X @ w)
        l = np.logaddexp(0.0, -m)
        if theta == 0:
            p = s
        else:
            z = theta * l + ls
            p = np.exp(z - z.max())
            p /= p.sum()
        a = -(y * sig(-m))[:, None] * X                     # per-row gradient of l_i, (n, d)
        g0v = p @ a
        g = g0v + lam * w
        if np.linalg.norm(g) < tol:
            break
        h = sig(m) * sig(-m)
        H = (X.T * (p * h)) @ X + lam * np.eye(d)
        if theta != 0:
            A = a - g0v
            H += theta * (A.T * p) @ A
        step = np.linalg.solve(H, g)
        t, f0 = 1.0, F(w)
        while F(w - t * step) > f0 - 0.25 * t * g @ step and t > 1e-12:
            t *= 0.5
        w = w - t * step
    return w


def group_dro(X, y, member, rounds=40, eta=2.0):
    """Minimax over group risks by multiplicative weights; returns the averaged-weights solution. member: (G, n) bool."""
    n = X.shape[0]
    G = member.shape[0]
    sizes = member.sum(1)
    q = np.ones(G) / G
    qbar = np.zeros(G)
    w = np.zeros(X.shape[1])
    for _ in range(rounds):
        s = (q / np.maximum(sizes, 1)) @ member + 1e-12
        w = tilted_solve(X, y, s, 0.0, w0=w)
        l = np.logaddexp(0.0, -y * (X @ w))
        risk = (member * l).sum(1) / np.maximum(sizes, 1)
        q = q * np.exp(eta * (risk - risk.max()))
        q /= q.sum()
        qbar += q
    qbar /= rounds
    s = (qbar / np.maximum(sizes, 1)) @ member + 1e-12
    return tilted_solve(X, y, s, 0.0, w0=w)


def evaluate(w, X, y, Xte, yte, tmember):
    pr = Xte @ w >= 0
    good = yte > 0
    err = pr != good
    fnr = float(err[good].mean())
    fpr = float(err[~good].mean())
    out = dict(acc=float(1 - err.mean()), train_acc=float(((X @ w >= 0) == (y > 0)).mean()), fnr=fnr, fpr=fpr,
               worst_class_err=max(fnr, fpr))
    if tmember is not None:
        out["group_err_counts"] = {g: [int(err[mk].sum()), int(mk.sum())] for g, mk in tmember.items()}
    return out


def split_data(name, sp):
    if name == "german":
        Xtr, ytr, Xte, yte = le.load_data(split_seed=sp, n_test=300)
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        m = test_mask(yall, sp)
        raw_tr, raw_te = df[~m].reset_index(drop=True), df[m].reset_index(drop=True)
        return Xtr, ytr, Xte, yte, raw_tr, raw_te
    Xtr, ytr, Xte, yte = g0.prepare(name, split_seed=sp)
    return Xtr, ytr, Xte, yte, None, None


def run(name, splits=SPLITS):
    res = []
    for sp in range(splits):
        Xtr, ytr, Xte, yte, raw_tr, raw_te = split_data(name, sp)
        if raw_tr is not None:
            gm_tr = {g: f(raw_tr).to_numpy() for g, f in GROUPS.items()}
            gm_tr = {g: m for g, m in gm_tr.items() if m.sum() >= 30}
            gm_te = {g: f(raw_te).to_numpy() for g, f in GROUPS.items()}
        else:
            gm_tr = gm_te = None
        n = len(ytr)
        s = np.ones(n) / n
        cell = {"ERM": evaluate(tilted_solve(Xtr, ytr, s, 0.0), Xtr, ytr, Xte, yte, gm_te)}
        for th in THETAS:
            wt = tilted_solve(Xtr, ytr, s, th)
            cell[f"tilt theta={th:g}"] = evaluate(wt, Xtr, ytr, Xte, yte, gm_te)
            # reweighting control: positive-class weight chosen so TRAIN accuracy matches the tilted model's (no test data used)
            tgt = cell[f"tilt theta={th:g}"]["train_acc"]
            best, bd = None, 9.0
            for c in np.geomspace(0.25, 4.0, 33):
                sw = np.where(ytr > 0, c, 1.0)
                w = tilted_solve(Xtr, ytr, sw / sw.sum(), 0.0)
                d = abs(float(((Xtr @ w >= 0) == (ytr > 0)).mean()) - tgt)
                if d < bd:
                    best, bd, bw = c, d, w
            cell[f"reweight (matched to theta={th:g})"] = dict(evaluate(bw, Xtr, ytr, Xte, yte, gm_te), pos_weight=float(best))
        if gm_tr is not None:
            member = np.stack(list(gm_tr.values()))
            wd = group_dro(Xtr, ytr, member)
            cell["group-DRO (known groups)"] = evaluate(wd, Xtr, ytr, Xte, yte, gm_te)
        res.append(cell)
        print(name, "split", sp, {k: round(v["acc"], 3) for k, v in cell.items()}, flush=True)
    return dict(dataset=name, splits=splits, cells=res)


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"tilted_{name}.json").write_text(json.dumps(run(name), indent=2))
