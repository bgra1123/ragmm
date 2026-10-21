"""
Experiment C (PLAN.md section 11): large n, a fraction of a pass to two passes of extra budget.
Run from the repository root:  python research/risk_effect/large_data_experiment.py 5000 20000 46842
Output: results/large_<n>.json
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepmatch_experiment as sm  # noqa: E402
import risk_effect_experiment as re_  # noqa: E402
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402

LAM, BATCH = sm.LAM, sm.BATCH
SEEDS = 100
BUDGETS = (0.25, 0.5, 1.0, 2.0)
N_BOOT = 1000
sig = le.sigmoid
RULES = {"R1(theta=1)": ("R1", 1.0), "R1(theta=5)": ("R1", 5.0), "R3(theta=1)": ("R3", 1.0), "R3(theta=5)": ("R3", 5.0)}


def alpha_w(rule, theta, W, G, X, y, wt):
    """sm.alpha_step with per-model row weights wt (S, n), rows summing to 1 (zero weight = row not used)."""
    M = y * (W @ X.T)
    C = y * (G @ X.T)
    s_ = sig(-M)
    wg = LAM * np.sum(W * G, 1)
    r1, r2 = -wg, 0.5 * LAM * np.sum(G * G, 1)
    dreg = lambda a: r1 + 2 * r2 * a        # noqa: E731
    ddreg = 2 * r2
    S = len(W)
    lo, hi = np.full(S, 1e-4), np.full(S, 1.0 / LAM)
    logw = np.where(wt > 0, np.log(np.maximum(wt, 1e-300)), -np.inf)
    if rule == "LS":
        h1 = lambda a: (wt * sig(-(M - a[:, None] * C)) * C).sum(1) + dreg(a)                                     # noqa: E731
        h2 = lambda a: (wt * sig(-(M - a[:, None] * C)) * sig(M - a[:, None] * C) * C * C).sum(1) + ddreg           # noqa: E731
    elif rule == "R1":
        A = -s_ * C + wg[:, None]

        def pw(a):
            z = -theta * a[:, None] * A + logw
            p = np.exp(z - z.max(1, keepdims=True))
            return p / p.sum(1, keepdims=True)
        h1 = lambda a: -(pw(a) * A).sum(1)                                                                        # noqa: E731
        h2 = lambda a: theta * ((lambda p: (p * A * A).sum(1) - (p * A).sum(1) ** 2)(pw(a)))                       # noqa: E731
    else:
        f_old = np.logaddexp(0.0, -M)

        def q(a):
            z = theta * (np.logaddexp(0.0, -(M - a[:, None] * C)) - f_old) + logw
            p = np.exp(z - z.max(1, keepdims=True))
            return p / p.sum(1, keepdims=True)

        def h1(a):
            return (q(a) * sig(-(M - a[:, None] * C)) * C).sum(1) + dreg(a)

        def h2(a):
            p = q(a)
            sg = sig(-(M - a[:, None] * C))
            u1, u2 = sg * C, sg * sig(M - a[:, None] * C) * C * C
            return (p * u2).sum(1) + theta * ((p * u1 * u1).sum(1) - (p * u1).sum(1) ** 2) + ddreg
    return sm.newton_vec(h1, h2, lo, hi, iters=40)


def grad_w(W, X, y, wt):
    M = y * (W @ X.T)
    return -((sig(-M) * y * wt) @ X) + LAM * W


def traj(W0, X, y, wts, rule=None, theta=0.0, scale=None, schedule=None):
    W = W0.copy()
    alphas = []
    for t, wt in enumerate(wts):
        G = grad_w(W, X, y, wt)
        if schedule is not None:
            al = np.full(len(W), schedule[t])
        else:
            al = alpha_w(rule, theta, W, G, X, y, wt)
            if scale is not None:
                al = scale * al
        W = W - al[:, None] * G
        alphas.append(al)
    return W, np.array(alphas)


def disagreement(D):
    p = D.mean(0)
    return float(np.mean(2 * p * (1 - p)) * len(D) / (len(D) - 1))


def arm(W, W1, D1, Xte, yte, ex):
    D = sig(W @ Xte.T) >= 0.5
    good = yte > 0
    rb, ra = D1 == good[None], D == good[None]
    delta = np.logaddexp(0.0, -yte * (W @ Xte.T)) - np.logaddexp(0.0, -yte * (W1 @ Xte.T))
    per = dict(harm=np.quantile(delta, 0.95, axis=1), neg=(rb & ~ra).mean(1), pos=(~rb & ra).mean(1), wrong=(D != good[None]).mean(1),
               changed=(D != ex[None]).sum(1).astype(float))
    summ = dict(disagreement=disagreement(D), changed=float(per["changed"].mean()), wrong=float(per["wrong"].mean()),
                harm_p95=float(per["harm"].mean()), neg=float(per["neg"].mean()), pos=float(per["pos"].mean()))
    return summ, per, D


def boot(pa, Da, pb, Db, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    S = len(Da)
    acc = {k: [] for k in ("harm", "neg", "wrong", "changed", "disagreement")}
    for _ in range(n):
        i = rng.integers(0, S, S)
        for k in ("harm", "neg", "wrong", "changed"):
            acc[k].append(pa[k][i].mean() - pb[k][i].mean())
        acc["disagreement"].append(disagreement(Da[i]) - disagreement(Db[i]))
    return {k: [float(np.mean(v)), *[float(x) for x in np.quantile(v, [0.025, 0.975])]] for k, v in acc.items()}


def run(n_train):
    t0 = time.time()
    Xall, yall, Xte, yte = g0.prepare("adult")
    rng0 = np.random.default_rng(0)
    idx = rng0.choice(len(yall), size=min(n_train, len(yall)), replace=False)
    X, y = Xall[idx], yall[idx]
    n, d = X.shape
    L, _ = le.constants(X, LAM)
    wstar, _ = le.solve_exact(X, y, LAM)
    ex = sig(Xte @ wstar) >= 0.5
    out = dict(dataset="adult", n=n, d=d, test=len(yte), seeds=SEEDS, stage1={})
    for opt in ("SGD", "Adam"):
        idx1 = le.batch_indices(range(SEEDS), 2000, n, BATCH)
        W1 = le.run_gmm(X, y, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0] if opt == "SGD" else sm.oe.run_opt("Adam", X, y, idx1, 0.01)
        D1 = sig(W1 @ Xte.T) >= 0.5
        lr0 = 1 / L if opt == "SGD" else 0.01
        s1, _, _ = arm(W1, W1, D1, Xte, yte, ex)
        res = {}
        for B in BUDGETS:
            if B >= 1:
                wts = [np.full((SEEDS, n), 1.0 / n) for _ in range(int(B))]
            else:
                m = max(1, round(B * n))
                rng = np.random.default_rng(int(B * 1000) + 7)
                mask = np.zeros((SEEDS, n))
                for s in range(SEEDS):
                    mask[s, rng.choice(n, m, replace=False)] = 1.0 / m
                wts = [mask]
            F, P, DD = {}, {}, {}
            K2 = max(1, round(B * n / BATCH))
            idx2 = le.batch_indices(range(100_000, 100_000 + SEEDS), K2, n, BATCH)
            for nm, mode in (("C1", "const"), ("C2", "decay"), ("C3", "avg")):
                W = re_.continue_training(W1, X, y, idx2, opt, lr0, mode)
                F[nm], P[nm], DD[nm] = arm(W, W1, D1, Xte, yte, ex)
            WL, aL = traj(W1, X, y, wts, "LS")
            F["L"], P["L"], DD["L"] = arm(WL, W1, D1, Xte, yte, ex)
            F["L"]["mean_alpha_times_L"] = float(aL.mean() * L)
            for key, (rule, th) in RULES.items():
                WR, aR = traj(W1, X, y, wts, rule, th)
                F[key], P[key], DD[key] = arm(WR, W1, D1, Xte, yte, ex)
                F[key]["mean_alpha_times_L"] = float(aR.mean() * L)
                if th == 5.0:
                    c = float(aR.mean() / aL.mean())
                    W_, _ = traj(W1, X, y, wts, "LS", scale=c)
                    F[f"M1[{key}]"], P[f"M1[{key}]"], DD[f"M1[{key}]"] = arm(W_, W1, D1, Xte, yte, ex)
                    W_, _ = traj(W1, X, y, wts, schedule=aR.mean(1))
                    F[f"M2[{key}]"], P[f"M2[{key}]"], DD[f"M2[{key}]"] = arm(W_, W1, D1, Xte, yte, ex)
            best = min(("C1", "C2", "C3"), key=lambda c_: F[c_]["disagreement"])
            tests = {}
            for key in RULES:
                refs = [best, "L"] + ([f"M1[{key}]", f"M2[{key}]"] if f"M1[{key}]" in F else [])
                for ref in refs:
                    tests[f"{key} vs {ref.split('[')[0] if ref.startswith('M') else ref}"] = boot(P[key], DD[key], P[ref], DD[ref])
            res[str(B)] = dict(best_continuation=best, arms=F, tests=tests)
            print(n, opt, B, f"[{time.time() - t0:.0f}s]", {k: round(v["disagreement"], 4) for k, v in F.items() if k in ("C1", "C3", "L", "R1(theta=1)", "R1(theta=5)")}, flush=True)
        out["stage1"][opt] = dict(stage1=s1, budgets=res)
    return out


if __name__ == "__main__":
    for n_ in (sys.argv[1:] or ["5000", "20000", "46842"]):
        (HERE / "results" / f"large_{n_}.json").write_text(json.dumps(run(int(n_)), indent=2))
