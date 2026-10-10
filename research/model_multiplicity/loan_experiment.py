"""
Experiment 1: do loan decisions depend on the training seed, and does RA-GMM help?

Pre-registered design: see PLAN.md in this folder.

Run from the repository root:
    python research/model_multiplicity/loan_experiment.py
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from risk_averse_tuning import GMM, RAGMM  # noqa: E402
from paper_tuning import ra_gmm_params  # noqa: E402

PROTECTED = ["personal_status_and_sex", "foreign_worker", "age_in_years"]
TARGET = "creditability"
COST_BAD_APPROVED = 5.0
COST_GOOD_REJECTED = 1.0
COST_THRESHOLD = COST_BAD_APPROVED / (COST_BAD_APPROVED + COST_GOOD_REJECTED)  # 5/6


# ---------------------------------------------------------------------- #
# Data                                                                     #
# ---------------------------------------------------------------------- #

def load_data(split_seed=0, n_test=300):
    df = pd.read_csv(HERE / "data" / "german_credit.csv")
    y = np.where(df[TARGET] == "good", 1.0, -1.0)
    feats = df.drop(columns=[TARGET] + PROTECTED)
    X = pd.get_dummies(feats, dtype=float).to_numpy()

    # Stratified split
    rng = np.random.default_rng(split_seed)
    test_idx = []
    for label in (1.0, -1.0):
        idx = np.flatnonzero(y == label)
        k = round(n_test * len(idx) / len(y))
        test_idx.extend(rng.choice(idx, size=k, replace=False))
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True

    Xtr, Xte = X[~test_mask], X[test_mask]
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    return Xtr, y[~test_mask], Xte, y[test_mask]


# ---------------------------------------------------------------------- #
# Objective                                                                #
# ---------------------------------------------------------------------- #

def log1pexp(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def objective(W, X, y, lam):
    """f(w) for each row of W (shape (S, d) or (d,))."""
    W2 = np.atleast_2d(W)
    margins = (W2 @ X.T) * y
    return log1pexp(-margins).mean(1) + 0.5 * lam * (W2 ** 2).sum(1)


def constants(X, lam):
    n = X.shape[0]
    L = np.linalg.eigvalsh(X.T @ X / n).max() / 4.0 + lam
    return float(L), float(lam)


def solve_exact(X, y, lam, tol=1e-10):
    """Newton's method with backtracking; returns w*, f*."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(100):
        m = y * (X @ w)
        s = sigmoid(-m)
        g = -(X.T @ (y * s)) / n + lam * w
        if np.linalg.norm(g) < tol:
            break
        h = s * (1 - s)
        H = (X.T * h) @ X / n + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        t, f0 = 1.0, objective(w, X, y, lam)[0]
        while objective(w - t * step, X, y, lam)[0] > f0 - 0.25 * t * g @ step:
            t *= 0.5
        w = w - t * step
    return w, objective(w, X, y, lam)[0]


# ---------------------------------------------------------------------- #
# Vectorized stochastic GMM (all seeds at once)                            #
# ---------------------------------------------------------------------- #

def batch_indices(seeds, K, n, b):
    """Minibatch index sequence per seed: shape (S, K, b). Same for every method."""
    return np.stack([
        np.random.default_rng(s).integers(0, n, size=(K, b)) for s in seeds
    ]).astype(np.int32)


def run_gmm(X, y, lam, idx, alpha, beta, gamma, tail_average=False, checkpoints=()):
    """
    GMM with stochastic gradients, vectorized over seeds:
        y_k = x_k + gamma d_k ;  d_{k+1} = beta d_k - alpha g(y_k) ;  x_{k+1} = x_k + d_{k+1}
    Returns final iterates (S, d) and the iterates at the requested checkpoints.
    """
    S, K, b = idx.shape
    d = X.shape[1]
    W = np.zeros((S, d))
    D = np.zeros((S, d))
    avg = np.zeros((S, d))
    n_avg = 0
    snaps = {}
    for k in range(K):
        Yk = W + gamma * D
        Xb = X[idx[:, k]]                       # (S, b, d)
        yb = y[idx[:, k]]                       # (S, b)
        m = np.einsum("sbd,sd->sb", Xb, Yk) * yb
        coef = -yb * sigmoid(-m)
        G = np.einsum("sbd,sb->sd", Xb, coef) / b + lam * Yk
        D = beta * D - alpha * G
        W = W + D
        if tail_average and k >= K // 2:
            n_avg += 1
            avg += (W - avg) / n_avg
        if (k + 1) in checkpoints:
            snaps[k + 1] = (avg if tail_average and n_avg else W).copy()
    return (avg if tail_average else W), snaps


def check_against_library(X, y, lam, alpha, beta, gamma, steps=50, seed=123):
    """Confirm the vectorized update matches the library's GMM optimizer (closure path)."""
    idx = batch_indices([seed], steps, X.shape[0], 16)
    W_vec, _ = run_gmm(X, y, lam, idx, alpha, beta, gamma)

    Xt, yt = torch.tensor(X), torch.tensor(y)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = GMM([w], lr=alpha, beta=beta, gamma=gamma)
    for k in range(steps):
        bi = torch.tensor(idx[0, k], dtype=torch.long)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.softplus(-(Xt[bi] @ w) * yt[bi]).mean() + 0.5 * lam * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
    err = np.abs(W_vec[0] - w.detach().numpy()).max()
    assert err < 1e-8, f"vectorized GMM deviates from library GMM: {err}"
    return err


# ---------------------------------------------------------------------- #
# Method configuration                                                     #
# ---------------------------------------------------------------------- #

def library_ragmm_params(L, mu, d, zeta):
    """What the repo's RAGMM (v0.1.0) picks. Recorded for transparency, not used as a method."""
    p = torch.zeros(d, requires_grad=True)
    gmm = GMM([p], lr=1.0 / (L + mu))
    ra = RAGMM(gmm, L=L, mu=mu, zeta=zeta)
    p.grad = torch.ones(d)          # tuning uses only the dimension d
    stats = ra.step()
    return {k: stats[k] for k in ("alpha", "beta", "gamma", "evar_bound")}


def tune_for_mean(X, y, lam, fstar, K, b, L, grid):
    """Pick the configuration with the lowest MEAN suboptimality on tuning seeds."""
    idx = batch_indices(range(10_000, 10_020), K, X.shape[0], b)
    best, best_val = None, float("inf")
    for cfg in grid:
        W, _ = run_gmm(X, y, lam, idx, **cfg)
        sub = objective(W, X, y, lam) - fstar
        val = np.nanmean(sub) if np.all(np.isfinite(sub)) else float("inf")
        if val < best_val:
            best, best_val = cfg, val
    return best


def build_methods(X, y, lam, fstar, K, b, L, mu, zeta, epsilons):
    kappa = L / mu
    m = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
    alphas = [c / L for c in np.logspace(-2, math.log10(2.0), 12)]
    methods = {
        "SGD (textbook)": dict(alpha=1 / L, beta=0.0, gamma=0.0),
        "Nesterov (textbook)": dict(alpha=1 / L, beta=m, gamma=m),
        "Heavy-ball (textbook)": dict(alpha=1 / L, beta=m, gamma=0.0),
    }
    methods["SGD (tuned for mean)"] = tune_for_mean(
        X, y, lam, fstar, K, b, L, [dict(alpha=a, beta=0.0, gamma=0.0) for a in alphas])
    methods["Nesterov (tuned for mean)"] = tune_for_mean(
        X, y, lam, fstar, K, b, L,
        [dict(alpha=a, beta=bt, gamma=bt) for a in alphas for bt in (0.5, 0.7, 0.8, 0.9, 0.95, m)])
    sgd_avg = tune_for_mean(
        X, y, lam, fstar, K, b, L,
        [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    methods["SGD + tail averaging (tuned)"] = sgd_avg
    ra_info = {}
    for eps in epsilons:
        p = ra_gmm_params(L, mu, X.shape[1], zeta, eps)
        name = f"RA-GMM (eps={eps})"
        methods[name] = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
        ra_info[name] = p
    ra_info["library RAGMM v0.1.0 (not used)"] = library_ragmm_params(L, mu, X.shape[1], zeta)
    return methods, ra_info


# ---------------------------------------------------------------------- #
# Metrics                                                                  #
# ---------------------------------------------------------------------- #

def tail_stats(v):
    v = np.asarray(v, dtype=float)
    q95 = np.quantile(v, 0.95)
    worst = np.sort(v)[-max(1, int(round(0.05 * len(v)))):]
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(q95),
                p99=float(np.quantile(v, 0.99)), cvar5=float(worst.mean()), max=float(v.max()))


def low_tail(v):
    """For 'higher is better' metrics: mean and the mean of the worst 5% (lowest)."""
    v = np.asarray(v, dtype=float)
    worst = np.sort(v)[:max(1, int(round(0.05 * len(v))))]
    return dict(mean=float(v.mean()), worst5_mean=float(worst.mean()), min=float(v.min()))


def auc(scores, labels):
    pos, neg = scores[labels > 0], scores[labels < 0]
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(W, Xtr, ytr, Xte, yte, lam, fstar, wstar, threshold):
    sub = objective(W, Xtr, ytr, lam) - fstar
    dist = np.linalg.norm(W - wstar, axis=1)
    P = sigmoid(W @ Xte.T)                         # (S, n_test) P(good)
    approve = P >= threshold
    ideal = sigmoid(Xte @ wstar) >= threshold
    flips = approve != ideal                        # (S, n_test)
    F = flips.sum(1)
    p_i = flips.mean(0)
    unanimous = approve.all(0) | (~approve).all(0)
    good = yte > 0
    acc = (approve == good).mean(1)
    cost = (COST_BAD_APPROVED * (approve & ~good) + COST_GOOD_REJECTED * (~approve & good)).mean(1)
    aucs = np.array([auc(P[s], yte) for s in range(len(W))])
    return dict(
        suboptimality=tail_stats(sub),
        dist_to_wstar=tail_stats(dist),
        flips_per_retrain=tail_stats(F),
        flip_prob_mean=float(p_i.mean()),
        applicants_flip_prob_ge_10pct=int((p_i >= 0.10).sum()),
        ambiguity=float(1 - unanimous.mean()),
        approval_rate_mean=float(approve.mean()),
        accuracy=low_tail(acc),
        cost_per_applicant=tail_stats(cost),
        auc=low_tail(aucs),
    ), dict(sub=sub, F=F, p_i=p_i, approve=approve)


def paired_bootstrap(a, b, stat, n_boot=10_000, seed=0):
    """CI for stat(a) - stat(b) with seeds resampled jointly."""
    rng = np.random.default_rng(seed)
    S = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, S, S)
        diffs[i] = stat(a[j]) - stat(b[j])
    return dict(diff=float(stat(a) - stat(b)),
                ci95=[float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))])


# ---------------------------------------------------------------------- #
# Main                                                                     #
# ---------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--K", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.05],
                    help="rate trade-off in problem (51); the first is the main RA-GMM setting")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[COST_THRESHOLD, 0.5])
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    Xtr, ytr, Xte, yte = load_data()
    n, d = Xtr.shape
    L, mu = constants(Xtr, args.lam)
    wstar, fstar = solve_exact(Xtr, ytr, args.lam)
    print(f"n={n} d={d} n_test={len(yte)} lambda={args.lam} L={L:.4f} mu={mu} kappa={L/mu:.1f} f*={fstar:.6f}")

    seeds = list(range(args.seeds))
    summary = dict(config=vars(args) | dict(n_train=n, n_test=int(len(yte)), d=d, L=L, mu=mu,
                                             kappa=L / mu, fstar=float(fstar)), runs={})
    for K in args.K:
        methods, ra_info = build_methods(Xtr, ytr, args.lam, fstar, K, args.batch, L, mu, args.zeta, args.eps)
        ref = methods[f"RA-GMM (eps={args.eps[0]})"]
        err = check_against_library(Xtr, ytr, args.lam, ref["alpha"], ref["beta"], ref["gamma"])
        idx = batch_indices(seeds, K, n, args.batch)
        checkpoints = sorted({c for c in np.unique(np.geomspace(10, K, 12).astype(int))} | {K})
        res_K = dict(methods={k: {kk: (float(vv) if not isinstance(vv, bool) else vv) for kk, vv in v.items()}
                              for k, v in methods.items()},
                     ragmm_tuning=ra_info, library_check_max_abs_err=float(err), by_threshold={},
                     curves={})
        raw = {}
        for name, cfg in methods.items():
            W, snaps = run_gmm(Xtr, ytr, args.lam, idx, checkpoints=set(checkpoints), **cfg)
            raw[name] = W
            res_K["curves"][name] = {int(c): tail_stats(objective(snaps[c], Xtr, ytr, args.lam) - fstar)
                                     for c in checkpoints if c in snaps}
        for thr in args.thresholds:
            key = f"{thr:.4f}"
            block, arrays = {}, {}
            for name, W in raw.items():
                block[name], arrays[name] = evaluate(W, Xtr, ytr, Xte, yte, args.lam, fstar, wstar, thr)
            ra_name = f"RA-GMM (eps={args.eps[0]})"
            comps = {}
            for name in raw:
                if name == ra_name:
                    continue
                comps[name] = dict(
                    flips_mean=paired_bootstrap(arrays[ra_name]["F"], arrays[name]["F"], np.mean),
                    flips_p95=paired_bootstrap(arrays[ra_name]["F"], arrays[name]["F"],
                                               lambda v: np.quantile(v, 0.95)),
                    subopt_p95=paired_bootstrap(arrays[ra_name]["sub"], arrays[name]["sub"],
                                                lambda v: np.quantile(v, 0.95)),
                )
            res_K["by_threshold"][key] = dict(metrics=block, ra_minus_baseline=comps)
            np.savez_compressed(out / f"arrays_lam{args.lam}_K{K}_thr{key}.npz",
                                **{f"{nm}|{a}": arrays[nm][a] for nm in arrays for a in ("F", "p_i", "sub")},
                                **{f"{nm}|approve": arrays[nm]["approve"] for nm in arrays})
        summary["runs"][str(K)] = res_K
        print(f"K={K} done ({time.time()-t0:.0f}s)")

    summary["ideal_model"] = dict(
        approval_rate={f"{t:.4f}": float((sigmoid(Xte @ wstar) >= t).mean()) for t in args.thresholds},
        test_auc=float(auc(sigmoid(Xte @ wstar), yte)))
    (out / f"summary_lam{args.lam}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/summary_lam{args.lam}.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
