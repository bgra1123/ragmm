"""
Risk-averse line search vs polishing baselines at equal compute. Plan: PLAN.md (pre-registered).
Run from the repository root:  python research/linesearch/linesearch_experiment.py german compas pima heart
"""
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "polish"))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import polish_experiments as pe  # noqa: E402
import run_gate0 as g0  # noqa: E402

LAM, BATCH, K1 = 0.01, 16, 2000
BUDGETS = (1, 5, 20)
THETAS = (1.0, 5.0)
GRID_POINTS = 80


def lme(z):
    """log mean exp along the last axis."""
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).mean(-1, keepdims=True)))[..., 0]


def grad_full(W, X, y):
    M = y * (W @ X.T)
    return -((le.sigmoid(-M) * y) @ X) / len(y) + LAM * W


def per_sample_loss(W, X, y):
    return np.logaddexp(0.0, -y * (W @ X.T)) + 0.5 * LAM * (W ** 2).sum(1, keepdims=True)


def step_size(rule, theta, W, G, X, y, L):
    """Vectorized over seeds. Returns alpha (S,), using a geometric grid then local refinement."""
    S = len(W)
    alphas = np.geomspace(1e-3 / L, 1.0 / (LAM), GRID_POINTS)         # up to 1/mu (mu = lambda): the largest useful line-search step
    M0 = y * (W @ X.T)                                             # (S, n)
    XG = G @ X.T                                                   # (S, n)  x_i^T g
    # per-sample first-order directional derivative a_i = g^T grad f_i(w)
    a = (-(le.sigmoid(-M0) * y) * XG) + LAM * (W * G).sum(1, keepdims=True)
    if rule == "R1":
        z = -theta * alphas[None, :, None] * a[:, None, :]          # (S, A, n)
        obj = lme(z) / theta
    else:
        Wa = W[:, None, :] - alphas[None, :, None] * G[:, None, :]  # (S, A, d)
        Ma = M0[:, None, :] - alphas[None, :, None] * (y * XG)[:, None, :]
        f_new = np.logaddexp(0.0, -Ma) + 0.5 * LAM * (Wa ** 2).sum(2, keepdims=True)
        if rule == "R3":
            f_old = per_sample_loss(W, X, y)[:, None, :]
            obj = lme(theta * (f_new - f_old)) / theta
        elif rule == "R4":
            obj = lme(theta * f_new) / theta
        elif rule == "LS":
            obj = f_new.mean(2)
        else:
            raise ValueError(rule)
    i = obj.argmin(1)
    # golden-section refinement in the bracket around the grid minimum (same objective, same rule)
    lo = alphas[np.clip(i - 1, 0, len(alphas) - 1)]
    hi = alphas[np.clip(i + 1, 0, len(alphas) - 1)]

    def phi(al):
        if rule == "R1":
            return lme(-theta * al[:, None] * a) / theta
        Wn = W - al[:, None] * G
        Mn = M0 - al[:, None] * (y * XG)
        fn = np.logaddexp(0.0, -Mn) + 0.5 * LAM * (Wn ** 2).sum(1, keepdims=True)
        if rule == "R3":
            return lme(theta * (fn - per_sample_loss(W, X, y))) / theta
        if rule == "R4":
            return lme(theta * fn) / theta
        return fn.mean(1)

    for _ in range(25):
        c, d = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        better = phi(c) < phi(d)
        hi = np.where(better, d, hi)
        lo = np.where(better, lo, c)
    return 0.5 * (lo + hi)


def gd(W0, X, y, T, rule, theta, L):
    W = W0.copy()
    alphas = []
    for _ in range(T):
        G = grad_full(W, X, y)
        if rule == "fixed":
            al = np.full(len(W), 1.0 / L)
        else:
            al = step_size(rule, theta, W, G, X, y, L)
        alphas.append(al)
        W = W - al[:, None] * G
    return W, np.mean(alphas, axis=0)


def metrics(W, W1, X, y, Xte, wstar, fstar):
    S = le.objective(W, X, y, LAM) - fstar
    q95 = np.quantile(S, 0.95)
    dec = le.sigmoid(W @ Xte.T) >= 0.5
    ex = le.sigmoid(Xte @ wstar) >= 0.5
    ch = (dec != ex[None]).sum(1)
    p = dec.mean(0)
    d_loss = per_sample_loss(W, X, y) - per_sample_loss(W1, X, y)
    f_new = per_sample_loss(W, X, y)
    return dict(S_mean=float(S.mean()), S_cvar5=float(S[S >= q95].mean()), changed=float(ch.mean()),
                disagreement=float(np.mean(2 * p * (1 - p)) * len(W) / (len(W) - 1)),
                share_rows_worse=float((d_loss > 0).mean()), p95_row_loss_change=float(np.quantile(d_loss, 0.95, axis=1).mean()),
                p95_row_loss=float(np.quantile(f_new, 0.95, axis=1).mean()), mean_row_loss=float(f_new.mean()))


def run(name, seeds=200):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    W1 = le.run_gmm(X, y, LAM, le.batch_indices(range(seeds), K1, n, BATCH), alpha=1 / L, beta=0.0, gamma=0.0)[0]
    out = dict(dataset=name, seeds=seeds, L=L, n=n, d=d, stage1=metrics(W1, W1, X, y, Xte, wstar, fstar), budgets={})
    rng = np.random.default_rng(0)
    for B in BUDGETS:
        K2 = max(1, round(B * n / BATCH))
        idx2 = le.batch_indices(range(100_000, 100_000 + seeds), K2, n, BATCH)
        cell = dict(K2=K2, methods={})
        sel, _ = pe.select_polish(X, y, W1, K2, rng)
        for nm, cfg in {"SGD decay (0.05/L)": dict(alpha=0.05 / L, beta=0.0, gamma=0.0),
                        "SGD average (1/L)": dict(alpha=1 / L, beta=0.0, gamma=0.0, average=True),
                        "adapter (exact EV@R)": sel}.items():
            W = pe.run_from(W1, X, y, idx2, **cfg)
            cell["methods"][nm] = metrics(W, W1, X, y, Xte, wstar, fstar)
        for nm, (rule, th) in {"GD fixed 1/L": ("fixed", 0), "GD exact line search (mean)": ("LS", 0),
                               **{f"R1 first-order risk θ={t:g}": ("R1", t) for t in THETAS},
                               **{f"R3 exact change risk θ={t:g}": ("R3", t) for t in THETAS},
                               **{f"R4 tilted loss risk θ={t:g}": ("R4", t) for t in THETAS}}.items():
            W, al = gd(W1, X, y, B, rule, th, L)
            cell["methods"][nm] = dict(metrics(W, W1, X, y, Xte, wstar, fstar), mean_alpha_times_L=float(al.mean() * L))
        out["budgets"][str(B)] = cell
        print(name, f"B={B}", {k.split(' (')[0][:22]: round(v['S_cvar5'], 5) for k, v in cell["methods"].items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"{name}.json").write_text(json.dumps(run(name), indent=2))
