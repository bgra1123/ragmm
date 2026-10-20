"""
Step-matched controls and theta sweep (PLAN.md section 3): is the risk-averse step better than simply taking a smaller step?
Run from the repository root:  python research/risk_effect/stepmatch_experiment.py german compas pima heart
Output: results/stepmatch_<dataset>.json

Arms (same stage-1 models, same full gradient direction at every step, budgets of 1, 2 and 5 full-batch steps):
  L          exact line search (mean loss)
  R1(th)     entropic first-order step (the original proposal), theta in {0.5, 1, 2, 5, 10}
  R3(th)     entropic exact-change step (variant), same theta grid
  M1(R)      matched control 1: the exact line-search step times one constant c, c = mean step of R / mean step of L (per dataset,
             budget and theta), so the average step length equals R's but the step keeps the line search's per-seed, per-iteration shape
  M2(R)      matched control 2: replay R's mean step length of each iteration (average over seeds) as one fixed schedule, same for all seeds
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
import optimizer_experiments as oe  # noqa: E402
import run_gate0 as g0  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

LAM, BATCH, K1, SEEDS = 0.01, 16, 2000, 200
BUDGETS = (1, 2, 5)
THETAS = (0.5, 1.0, 2.0, 5.0, 10.0)
N_BOOT = 1000
sig = le.sigmoid


def newton_vec(h1, h2, lo, hi, iters=60):
    """Safeguarded Newton for a convex scalar function per seed. h1, h2: alpha (S,) -> derivative values (S,)."""
    a, b = lo.copy(), hi.copy()
    at_lo = h1(lo) >= 0
    at_hi = h1(hi) <= 0
    x = np.sqrt(lo * hi)
    for _ in range(iters):
        d1 = h1(x)
        neg = d1 < 0
        a = np.where(neg, x, a)
        b = np.where(neg, b, x)
        d2 = h2(x)
        xn = np.where(d2 > 0, x - d1 / np.where(d2 > 0, d2, 1.0), 0.5 * (a + b))
        xn = np.where((xn > a) & (xn < b), xn, 0.5 * (a + b))
        if np.all(np.abs(xn - x) <= 1e-12 * np.maximum(x, 1e-30)):
            x = xn
            break
        x = xn
    return np.where(at_lo, lo, np.where(at_hi, hi, x))


def alpha_step(rule, theta, W, G, X, y):
    M = y * (W @ X.T)
    C = y * (G @ X.T)
    s_ = sig(-M)
    wg = LAM * np.sum(W * G, 1)
    r1 = -wg
    r2 = 0.5 * LAM * np.sum(G * G, 1)
    dreg = lambda a: r1 + 2 * r2 * a        # noqa: E731
    ddreg = 2 * r2
    S = len(W)
    lo, hi = np.full(S, 1e-4), np.full(S, 1.0 / LAM)
    if rule == "LS":
        h1 = lambda a: np.mean(sig(-(M - a[:, None] * C)) * C, 1) + dreg(a)                                   # noqa: E731
        h2 = lambda a: np.mean(sig(-(M - a[:, None] * C)) * sig(M - a[:, None] * C) * C * C, 1) + ddreg       # noqa: E731
    elif rule == "R1":
        A = -s_ * C + wg[:, None]

        def pw(a):
            z = -theta * a[:, None] * A
            p = np.exp(z - z.max(1, keepdims=True))
            return p / p.sum(1, keepdims=True)
        h1 = lambda a: -(pw(a) * A).sum(1)                                                                    # noqa: E731
        h2 = lambda a: theta * ((lambda p: (p * A * A).sum(1) - (p * A).sum(1) ** 2)(pw(a)))                   # noqa: E731
    else:  # R3
        f_old = np.logaddexp(0.0, -M)

        def q(a):
            z = theta * (np.logaddexp(0.0, -(M - a[:, None] * C)) - f_old)
            p = np.exp(z - z.max(1, keepdims=True))
            return p / p.sum(1, keepdims=True)

        def h1(a):
            return (q(a) * sig(-(M - a[:, None] * C)) * C).sum(1) + dreg(a)

        def h2(a):
            p = q(a)
            sg = sig(-(M - a[:, None] * C))
            u1, u2 = sg * C, sg * sig(M - a[:, None] * C) * C * C
            return (p * u2).sum(1) + theta * ((p * u1 * u1).sum(1) - (p * u1).sum(1) ** 2) + ddreg
    return newton_vec(h1, h2, lo, hi)


def grad_full(W, X, y):
    M = y * (W @ X.T)
    return -((sig(-M) * y) @ X) / len(y) + LAM * W


def trajectory(W0, X, y, T, rule=None, theta=0.0, scale=None, schedule=None):
    """Full-gradient descent for T steps. Returns {t: W_t} and the step lengths (T, S)."""
    W = W0.copy()
    snaps, alphas = {}, []
    for t in range(T):
        G = grad_full(W, X, y)
        if schedule is not None:
            al = np.full(len(W), schedule[t])
        else:
            al = alpha_step(rule, theta, W, G, X, y)
            if scale is not None:
                al = scale * al
        W = W - al[:, None] * G
        alphas.append(al)
        snaps[t + 1] = W.copy()
    return snaps, np.array(alphas)


def stage1(opt, X, y, L):
    idx = le.batch_indices(range(SEEDS), K1, X.shape[0], BATCH)
    if opt == "SGD":
        return le.run_gmm(X, y, LAM, idx, alpha=1 / L, beta=0.0, gamma=0.0)[0]
    return oe.run_opt("Adam", X, y, idx, 0.01)


def disagreement(D):
    p = D.mean(0)
    return float(np.mean(2 * p * (1 - p)) * len(D) / (len(D) - 1))


def arm_stats(W, W1, X, y, Xte, yte, ex, fstar, groups):
    D = sig(W @ Xte.T) >= 0.5
    good = yte > 0
    delta_te = np.logaddexp(0.0, -yte * (W @ Xte.T)) - np.logaddexp(0.0, -yte * (W1 @ Xte.T))
    delta_tr = np.logaddexp(0.0, -y * (W @ X.T)) - np.logaddexp(0.0, -y * (W1 @ X.T))
    per = dict(harm_te=np.quantile(delta_te, 0.95, axis=1), harm_tr=np.quantile(delta_tr, 0.95, axis=1),
               worse_te=(delta_te > 0).mean(1), wrong=(D != good[None]).mean(1),
               gap=le.objective(W, X, y, LAM) - fstar)
    if groups:
        per["group_wrong"] = np.stack([(D[:, mk] != good[mk][None]).mean(1) for mk in groups.values()], 1)
    ch = (D != ex[None]).sum(1)
    summ = dict(disagreement=disagreement(D), changed_mean=float(ch.mean()), harm_te_p95=float(per["harm_te"].mean()),
                harm_tr_p95=float(per["harm_tr"].mean()), share_rows_worse_te=float(per["worse_te"].mean()),
                wrong=float(per["wrong"].mean()), gap_mean=float(per["gap"].mean()),
                gap_cvar5=float(np.sort(per["gap"])[-max(1, len(per["gap"]) // 20):].mean()))
    if groups:
        summ["worst_group_wrong"] = float(per["group_wrong"].mean(0).max())
    return summ, per, D


def boot_pair(pa, pb, Da, Db, groups, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    S = len(Da)
    out = {k: [] for k in ("harm_te", "harm_tr", "wrong", "disagreement", "worst_group")}
    for _ in range(n):
        i = rng.integers(0, S, S)
        out["harm_te"].append(pa["harm_te"][i].mean() - pb["harm_te"][i].mean())
        out["harm_tr"].append(pa["harm_tr"][i].mean() - pb["harm_tr"][i].mean())
        out["wrong"].append(pa["wrong"][i].mean() - pb["wrong"][i].mean())
        out["disagreement"].append(disagreement(Da[i]) - disagreement(Db[i]))
        if groups:
            out["worst_group"].append(pa["group_wrong"][i].mean(0).max() - pb["group_wrong"][i].mean(0).max())
    res = {}
    for k, v in out.items():
        if v:
            res[k] = [float(np.mean(v)), *[float(x) for x in np.quantile(v, [0.025, 0.975])]]
    return res


def run(name):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    ex = sig(Xte @ wstar) >= 0.5
    groups = None
    if name == "german":
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        raw = df[test_mask(yall, 0)].reset_index(drop=True)
        groups = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
        groups = {g: mk for g, mk in groups.items() if mk.sum() >= 30}
    out = dict(dataset=name, seeds=SEEDS, n=n, groups=list(groups) if groups else None, stage1={})
    Tmax = max(BUDGETS)
    for opt in ("SGD", "Adam"):
        W1 = stage1(opt, X, y, L)
        res = {}
        s1, p1, D1 = arm_stats(W1, W1, X, y, Xte, yte, ex, fstar, groups)
        base = {B: {} for B in BUDGETS}
        store = {B: {} for B in BUDGETS}           # per-seed arrays and decisions for the bootstrap
        snaps_L, al_L = trajectory(W1, X, y, Tmax, "LS")
        R_runs = {}
        for rule in ("R1", "R3"):
            for th in THETAS:
                R_runs[(rule, th)] = trajectory(W1, X, y, Tmax, rule, th)
        for B in BUDGETS:
            s, p, D = arm_stats(snaps_L[B], W1, X, y, Xte, yte, ex, fstar, groups)
            s["mean_alpha_times_L"] = float(al_L[:B].mean() * L)
            base[B]["L"] = s
            store[B]["L"] = (p, D)
            for (rule, th), (snaps, al) in R_runs.items():
                key = f"{rule}(theta={th:g})"
                s, p, D = arm_stats(snaps[B], W1, X, y, Xte, yte, ex, fstar, groups)
                s["mean_alpha_times_L"] = float(al[:B].mean() * L)
                base[B][key], store[B][key] = s, (p, D)
                # matched control 1: scaled exact line search
                c = float(al[:B].mean() / al_L[:B].mean())
                sn, _ = trajectory(W1, X, y, B, "LS", scale=c)
                s, p, D = arm_stats(sn[B], W1, X, y, Xte, yte, ex, fstar, groups)
                s["mean_alpha_times_L"] = float(c * al_L[:B].mean() * L)
                s["c"] = c
                base[B][f"M1[{key}]"], store[B][f"M1[{key}]"] = s, (p, D)
                # matched control 2: replay R's per-iteration mean step as one schedule
                sn, _ = trajectory(W1, X, y, B, schedule=al[:B].mean(1))
                s, p, D = arm_stats(sn[B], W1, X, y, Xte, yte, ex, fstar, groups)
                s["mean_alpha_times_L"] = float(al[:B].mean() * L)
                base[B][f"M2[{key}]"], store[B][f"M2[{key}]"] = s, (p, D)
            tests = {}
            for key in [k for k in base[B] if k.startswith("R")]:
                for ref in ("L", f"M1[{key}]", f"M2[{key}]"):
                    pa, Da = store[B][key]
                    pb, Db = store[B][ref]
                    tests[f"{key} vs {ref.split('[')[0]}"] = boot_pair(pa, pb, Da, Db, groups)
            res[str(B)] = dict(arms=base[B], tests=tests)
            print(name, opt, f"B={B}", f"[{time.time() - t0:.0f}s]", flush=True)
        out["stage1"][opt] = dict(stage1=s1, budgets=res)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"stepmatch_{name}.json").write_text(json.dumps(run(name), indent=2))
