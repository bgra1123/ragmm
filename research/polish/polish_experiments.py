"""
E1 (fixed learning rate) and E2 (polish any solver). Design: PLAN.md (pre-registered).
Run from the repository root:
    python research/polish/polish_experiments.py e1 --datasets german compas pima heart adult bank
    python research/polish/polish_experiments.py e2 --datasets german compas pima heart
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from risk_averse_tuning import RAGMMBounds  # noqa: E402
from risk_averse_tuning.quadratic import evar_quadratic  # noqa: E402
from risk_averse_tuning.tail_report import empirical_evar  # noqa: E402

LAM, BATCH, ZETA = 0.01, 16, 0.05
BETAS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)


# ---------------------------------------------------------------------- #
# Vectorized stochastic GMM from arbitrary starting points                 #
# ---------------------------------------------------------------------- #

def run_from(W0, X, y, idx, alpha, beta, gamma, average=False):
    S, K, b = idx.shape
    W = W0.copy()
    D = np.zeros_like(W)
    avg, n_avg = np.zeros_like(W), 0
    for k in range(K):
        Y = W + gamma * D
        Xb, yb = X[idx[:, k]], y[idx[:, k]]
        coef = -yb * le.sigmoid(-np.einsum("sbd,sd->sb", Xb, Y) * yb)
        G = np.einsum("sbd,sb->sd", Xb, coef) / b + LAM * Y
        D = beta * D - alpha * G
        W = W + D
        if average:
            n_avg += 1
            avg += (W - avg) / n_avg
    return avg if average else W


def hessian(X, y, w):
    s = le.sigmoid(y * (X @ w))
    return (X.T * (s * (1 - s))) @ X / X.shape[0] + LAM * np.eye(X.shape[1])


def grad_cov(X, y, w):
    """Covariance of the per-row gradient (population, over rows), divided by the batch size."""
    g = -(y * le.sigmoid(-y * (X @ w)))[:, None] * X
    g = g - g.mean(0)
    return g.T @ g / len(g) / BATCH


def metrics(W, X, y, Xte, yte, wstar, fstar):
    S = le.objective(W, X, y, LAM) - fstar
    ok = np.isfinite(S)
    if not ok.all():
        return dict(diverged=int((~ok).sum()))
    q95 = float(np.quantile(S, 0.95))
    tail = S >= q95
    P = le.sigmoid(W @ Xte.T) >= 0.5
    ex = le.sigmoid(Xte @ wstar) >= 0.5
    ch = (P != ex[None]).sum(1)
    share = P.mean(0)
    k5 = max(1, len(S) // 20)
    return dict(S_mean=float(S.mean()), S_p95=q95, S_cvar5=float(S[tail].mean()), S_evar=empirical_evar(S, ZETA),
                changed_mean=float(ch.mean()), changed_worst5=float(np.sort(ch)[-k5:].mean()),
                coin_flip=int((np.minimum(share, 1 - share) >= 0.10).sum()),
                accuracy=float(((P == (yte > 0)[None]).mean(1)).mean())), S


def boot_diff(a, b, stat, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = [stat(a[j]) - stat(b[j]) for j in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return dict(diff=float(stat(a) - stat(b)), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])


def cvar5(v):
    return float(np.sort(v)[-max(1, len(v) // 20):].mean())


# ---------------------------------------------------------------------- #
# E1                                                                       #
# ---------------------------------------------------------------------- #

def e1(name, seeds, K=2000):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    H, Sig = hessian(X, y, wstar), grad_cov(X, y, wstar)
    idx = le.batch_indices(range(seeds), K, n, BATCH)
    W0 = np.zeros((seeds, d))
    bounds = RAGMMBounds(zeta=ZETA)
    out = dict(dataset=name, L=L, kappa=L / mu, cells={})
    # reference: careful SGD
    alphas = [c / L for c in np.logspace(-2, math.log10(2.0), 12)]
    careful = le.tune_for_mean(X, y, LAM, fstar, K, BATCH, L,
                               [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    Wc, _ = le.run_gmm(X, y, LAM, idx, **careful)
    out["reference_careful_sgd"] = metrics(Wc, X, y, Xte, yte, wstar, fstar)[0]
    for c in (1.0, 0.3, 0.1, 0.03):
        alpha = c / L
        sol = bounds.solve_fixed_alpha(alpha, L, mu, d)
        m_agd = (1 - math.sqrt(alpha * mu)) / (1 + math.sqrt(alpha * mu))
        best = None
        for b in BETAS:
            for g in sorted({0.0, b / 2, b}):
                v = evar_quadratic(H, alpha, b, g, K, Sig, -wstar, ZETA)
                if math.isfinite(v) and (best is None or v < best[0]):
                    best = (v, b, g)
        eff = min(alpha / (1 - sol.beta), 1.0 / L)
        methods = {
            "SGD": dict(alpha=alpha, beta=0.0, gamma=0.0),
            "Nesterov (textbook momentum)": dict(alpha=alpha, beta=m_agd, gamma=m_agd),
            "RA-GMM at alpha": dict(alpha=alpha, beta=sol.beta, gamma=sol.gamma),
            "Exact-EV@R at alpha (oracle)": dict(alpha=alpha, beta=best[1], gamma=best[2]),
            "SGD at effective step alpha/(1-beta_RA)": dict(alpha=eff, beta=0.0, gamma=0.0),
        }
        cell = dict(alpha_over_invL=c, params={k: {a: float(v) for a, v in cfg.items()} for k, cfg in methods.items()},
                    ra_region=sol.region, ra_psi=sol.psi, methods={}, paired={})
        Ss = {}
        for nm, cfg in methods.items():
            W = run_from(W0, X, y, idx, **cfg)
            r = metrics(W, X, y, Xte, yte, wstar, fstar)
            if isinstance(r, dict):
                cell["methods"][nm] = r
            else:
                cell["methods"][nm], Ss[nm] = r
        ra = "RA-GMM at alpha"
        for other in ("SGD", "Nesterov (textbook momentum)", "Exact-EV@R at alpha (oracle)",
                      "SGD at effective step alpha/(1-beta_RA)"):
            if ra in Ss and other in Ss:
                cell["paired"][f"RA-GMM at alpha minus {other}"] = dict(cvar5=boot_diff(Ss[ra], Ss[other], cvar5),
                                                                          mean=boot_diff(Ss[ra], Ss[other], np.mean))
        if "Exact-EV@R at alpha (oracle)" in Ss and "SGD" in Ss:
            cell["paired"]["Exact minus SGD"] = dict(cvar5=boot_diff(Ss["Exact-EV@R at alpha (oracle)"], Ss["SGD"], cvar5))
        out["cells"][f"{c}"] = cell
        print(f"{name} alpha={c}/L " + " | ".join(
            f"{k.split(' ')[0]}{'-eff' if 'effective' in k else ''}: cvar {v.get('S_cvar5', float('nan')):.4f} ch {v.get('changed_mean', float('nan')):.1f}"
            for k, v in cell["methods"].items()) + f" [{time.time() - t0:.0f}s]", flush=True)
    return out


# ---------------------------------------------------------------------- #
# E2                                                                       #
# ---------------------------------------------------------------------- #

def admm(X, y, rng, iters=5, blocks=4, rho=0.25, newton=2):
    n, d = X.shape
    part = np.array_split(rng.permutation(n), blocks)
    z = np.zeros(d)
    w = np.zeros((blocks, d))
    u = np.zeros((blocks, d))
    for _ in range(iters):
        for b, rows in enumerate(part):
            Xb, yb = X[rows], y[rows]
            for _ in range(newton):
                s = le.sigmoid(yb * (Xb @ w[b]))
                g = -(Xb.T @ (yb * (1 - s))) / n + rho * (w[b] - z + u[b])
                Hb = (Xb.T * (s * (1 - s))) @ Xb / n + rho * np.eye(d)
                w[b] = w[b] - np.linalg.solve(Hb, g)
        z = blocks * rho * (w + u).mean(0) / (LAM + blocks * rho)
        u += w - z
    return z


def select_polish(X, y, W1, K2, rng, pilots=6):
    L = le.constants(X, LAM)[0]
    pil = W1[rng.choice(len(W1), pilots, replace=False)]
    wbar = pil.mean(0)
    H, Sig = hessian(X, y, wbar), grad_cov(X, y, wbar)
    e0s = []
    for w1 in pil:
        g = -(X.T @ (y * le.sigmoid(-y * (X @ w1)))) / len(y) + LAM * w1
        e0s.append(-np.linalg.solve(hessian(X, y, w1), g))
    best = None
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        for b in (0.0, 0.5, 0.9, 0.98):
            for g in sorted({0.0, b}):
                v = np.mean([evar_quadratic(H, c / L, b, g, K2, Sig, e0, ZETA) for e0 in e0s])
                if math.isfinite(v) and (best is None or v < best[0]):
                    best = (v, c / L, b, g)
    return dict(alpha=best[1], beta=best[2], gamma=best[3]), best[0]


def e2(name, seeds, K1=2000):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    out = dict(dataset=name, stages={})
    idx1 = le.batch_indices(range(seeds), K1, n, BATCH)
    stage1 = {"SGD (textbook, 2000 steps)": le.run_gmm(X, y, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0],
              "ADMM (5 iterations)": np.stack([admm(X, y, np.random.default_rng(s)) for s in range(seeds)])}
    ra = RAGMMBounds(zeta=ZETA).solve(L, mu, d, rate_slack=0.25)
    rng = np.random.default_rng(0)
    for st, W1 in stage1.items():
        res1 = metrics(W1, X, y, Xte, yte, wstar, fstar)
        out["stages"][st] = dict(stage1=res1[0] if not isinstance(res1, dict) else res1, K2={})
        for K2 in (200, 1000):
            idx2 = le.batch_indices(range(100_000, 100_000 + seeds), K2, n, BATCH)
            sel, sel_val = select_polish(X, y, W1, K2, rng)
            polishers = {
                "none": None,
                "continue (SGD, 1/L)": dict(alpha=1 / L, beta=0.0, gamma=0.0),
                "decay (SGD, 0.05/L)": dict(alpha=0.05 / L, beta=0.0, gamma=0.0),
                "average (SGD, 1/L, averaged)": dict(alpha=1 / L, beta=0.0, gamma=0.0, average=True),
                "RA-GMM (51)": dict(alpha=ra.alpha, beta=ra.beta, gamma=ra.gamma),
                "Exact-EV@R polish": sel,
            }
            cell = dict(selected={k: {a: float(v) for a, v in c.items() if a != "average"} for k, c in polishers.items() if c},
                        methods={}, paired={})
            Ss = {}
            for nm, cfg in polishers.items():
                W = W1 if cfg is None else run_from(W1, X, y, idx2, **cfg)
                r = metrics(W, X, y, Xte, yte, wstar, fstar)
                if isinstance(r, dict):
                    cell["methods"][nm] = r
                else:
                    cell["methods"][nm], Ss[nm] = r
            for nm in Ss:
                if nm != "none":
                    cell["methods"][nm]["share_seeds_improved"] = float(np.mean(Ss[nm] < Ss["none"]))
                    cell["paired"][f"{nm} minus none"] = dict(cvar5=boot_diff(Ss[nm], Ss["none"], cvar5))
            out["stages"][st]["K2"][str(K2)] = cell
            print(f"{name} | {st} | K2={K2} " + " | ".join(
                f"{k.split(' ')[0]}: cvar {v.get('S_cvar5', float('nan')):.4f} ch {v.get('changed_mean', float('nan')):.1f}"
                for k, v in cell["methods"].items()) + f" [{time.time() - t0:.0f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2"])
    ap.add_argument("--datasets", nargs="+", default=["german", "compas", "pima", "heart"])
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    for name in args.datasets:
        seeds = args.seeds or (400 if args.exp == "e1" else 200)
        res = e1(name, seeds) if args.exp == "e1" else e2(name, seeds)
        (HERE / "results" / f"{args.exp}_{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
