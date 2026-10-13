"""
Which data rows carry the loss of the worst-5% retrained models? Design: PLAN.md (pre-registered).
Run from the repository root:  python research/tail_analysis/tail_rows.py [--only pima ...]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
from paper_tuning import ra_gmm_params  # noqa: E402

LAM, BATCH, K, SEEDS, ZETA = 0.01, 16, 2000, 1000, 0.05


def raw_frame(name):
    """Original features (before one-hot / scaling), label in {1,-1}, and the train mask used by Gate 0."""
    if name == "german":
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        y = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        Xdf = df.drop(columns=[le.TARGET] + le.PROTECTED)
        n_test = 300
    else:
        Xdf, y01 = g0.DATASETS[name]()
        y = np.where(np.asarray(y01) == 1, 1.0, -1.0)
        n_test = min(2000, int(round(0.3 * len(y))))
    rng = np.random.default_rng(0)
    test_idx = []
    for label in (1.0, -1.0):
        idx = np.flatnonzero(y == label)
        test_idx.extend(rng.choice(idx, size=round(n_test * len(idx) / len(y)), replace=False))
    mask = np.zeros(len(y), bool)
    mask[test_idx] = True
    return Xdf.reset_index(drop=True), y, ~mask


def row_losses(W, X, y):
    """ℓ_i(w) for each model (rows of W) and data row: shape (models, rows)."""
    return np.logaddexp(0.0, -(W @ X.T) * y)


def matched_alpha(X, y, fstar, target_mean, L):
    idx = le.batch_indices(range(10_000, 10_100), K, X.shape[0], BATCH)

    def mean_S(a):
        W, _ = le.run_gmm(X, y, LAM, idx, a, 0.0, 0.0)
        return float(np.mean(le.objective(W, X, y, LAM) - fstar))

    grid = np.geomspace(1e-3 / L, 1.0 / L, 13)
    vals = np.array([mean_S(a) for a in grid])
    lo_i = int(np.argmin(vals))
    if vals[lo_i] > target_mean:
        return float(grid[lo_i]), False
    lo, hi = np.log(grid[lo_i]), np.log(1.0 / L)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if mean_S(np.exp(mid)) < target_mean:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi))), True


def profile(top, rest, margin, lev, rarity, y, D, cols):
    smd = (D[top].mean(0) - D[rest].mean(0)) / np.sqrt(0.5 * (D[top].var(0) + D[rest].var(0)) + 1e-12)
    order = np.argsort(-np.abs(smd))[:5]
    return dict(
        n_rows=int(top.sum()),
        median_margin_top=float(np.median(margin[top])), median_margin_rest=float(np.median(margin[rest])),
        misclassified_by_exact_top=float(np.mean(margin[top] < 0)), misclassified_by_exact_rest=float(np.mean(margin[rest] < 0)),
        median_leverage_ratio=float(np.median(lev[top]) / np.median(lev[rest])),
        median_rarity_top=float(np.median(rarity[top])), median_rarity_rest=float(np.median(rarity[rest])),
        positive_share_top=float(np.mean(y[top] > 0)), positive_share_rest=float(np.mean(y[rest] > 0)),
        top_features=[dict(feature=cols[j], smd=float(smd[j]), mean_top=float(D[top, j].mean()),
                           mean_rest=float(D[rest, j].mean())) for j in order])


def run(name):
    t0 = time.time()
    Xtr, ytr, Xte, yte = g0.prepare(name)
    Xdf, yall, train_mask = raw_frame(name)
    assert train_mask.sum() == len(ytr) and np.array_equal(yall[train_mask], ytr)
    raw = Xdf[train_mask].reset_index(drop=True)
    cat_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    D = pd.get_dummies(raw, dtype=float)
    cols = list(D.columns)
    Dn = D.to_numpy()
    rarity = np.ones(len(raw))
    for c in cat_cols:
        freq = raw[c].map(raw[c].value_counts(normalize=True)).to_numpy()
        rarity = np.minimum(rarity, freq)

    n, d = Xtr.shape
    L, mu = le.constants(Xtr, LAM)
    wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
    m_star = ytr * (Xtr @ wstar)
    s = le.sigmoid(m_star)
    H = (Xtr.T * (s * (1 - s))) @ Xtr / n + LAM * np.eye(d)
    lev = s * (1 - s) * np.einsum("ij,ij->i", Xtr @ np.linalg.inv(H), Xtr) / n
    loss_star = np.logaddexp(0.0, -m_star)

    p = ra_gmm_params(L, mu, d, ZETA, 0.25)
    ra = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
    idx_t = le.batch_indices(range(10_000, 10_100), K, n, BATCH)
    Wt, _ = le.run_gmm(Xtr, ytr, LAM, idx_t, **ra)
    ra_mean = float(np.mean(le.objective(Wt, Xtr, ytr, LAM) - fstar))
    a_match, matched = matched_alpha(Xtr, ytr, fstar, ra_mean, L)
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    tuned = le.tune_for_mean(Xtr, ytr, LAM, fstar, K, BATCH, L,
                             [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    methods = {"SGD (textbook)": dict(alpha=1 / L, beta=0.0, gamma=0.0), "RA-GMM (eps=0.25)": ra,
               "SGD (matched mean)": dict(alpha=a_match, beta=0.0, gamma=0.0),
               "SGD + tail averaging (tuned)": tuned}

    idx = le.batch_indices(range(SEEDS), K, n, BATCH)
    out = dict(dataset=name, n_train=n, d=d, kappa=L / mu, matched_mean_found=matched,
               params={k: {kk: float(v) for kk, v in c.items()} for k, c in methods.items()}, methods={})
    tops, Xs = {}, {}
    for nm, cfg in methods.items():
        W, _ = le.run_gmm(Xtr, ytr, LAM, idx, **cfg)
        S = le.objective(W, Xtr, ytr, LAM) - fstar
        q95 = np.quantile(S, 0.95)
        tail, body = S > q95, S <= q95
        X_tail = (row_losses(W[tail], Xtr, ytr) - loss_star).mean(0)
        X_body = np.zeros(n)
        for chunk in np.array_split(np.flatnonzero(body), 20):
            X_body += (row_losses(W[chunk], Xtr, ytr) - loss_star).sum(0)
        X_body /= body.sum()
        k = max(1, int(round(0.05 * n)))
        top = np.zeros(n, bool)
        top[np.argsort(-X_tail)[:k]] = True
        pos = np.clip(X_tail, 0, None)
        tops[nm], Xs[nm] = top, X_tail
        top_body = np.zeros(n, bool)
        top_body[np.argsort(-X_body)[:k]] = True
        out["methods"][nm] = dict(
            S_mean=float(S.mean()), S_p95=float(q95), S_cvar5=float(S[tail].mean()),
            S_cvar5_over_mean=float(S[tail].mean() / S.mean()),
            tail_excess_loss_mean=float(X_tail.mean()), body_excess_loss_mean=float(X_body.mean()),
            concentration_top5pct_rows=float(pos[top].sum() / pos.sum()),
            spearman_tail_vs_body=float(spearmanr(X_tail, X_body)[0]),
            top_rows_overlap_tail_vs_body=float((top & top_body).sum() / k),
            spearman_with_leverage=float(spearmanr(X_tail, lev)[0]),
            spearman_with_abs_margin=float(spearmanr(X_tail, np.abs(m_star))[0]),
            spearman_with_margin=float(spearmanr(X_tail, m_star)[0]),
            spearman_with_rarity=float(spearmanr(X_tail, rarity)[0]) if cat_cols else None,
            profile=profile(top, ~top, m_star, lev, rarity, ytr, Dn, cols))
    a, b = tops["SGD (textbook)"], tops["RA-GMM (eps=0.25)"]
    c = tops["SGD (matched mean)"]
    out["overlap"] = dict(
        jaccard_sgd_textbook_vs_ragmm=float((a & b).sum() / (a | b).sum()),
        jaccard_sgd_matched_vs_ragmm=float((c & b).sum() / (c | b).sum()),
        ragmm_over_sgd_textbook_excess_on_sgd_top_rows=float(Xs["RA-GMM (eps=0.25)"][a].mean() / Xs["SGD (textbook)"][a].mean()),
        ragmm_over_sgd_matched_excess_on_ragmm_top_rows=float(Xs["RA-GMM (eps=0.25)"][b].mean() / Xs["SGD (matched mean)"][b].mean()),
        ragmm_over_sgd_matched_excess_all_rows=float(Xs["RA-GMM (eps=0.25)"].mean() / Xs["SGD (matched mean)"].mean()),
        ragmm_over_sgd_matched_excess_on_union_top_rows=float(Xs["RA-GMM (eps=0.25)"][b | c].mean() / Xs["SGD (matched mean)"][b | c].mean()),
        ragmm_over_sgd_textbook_excess_all_rows=float(Xs["RA-GMM (eps=0.25)"].mean() / Xs["SGD (textbook)"].mean()))
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(g0.DATASETS))
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    for name in args.only:
        r = run(name)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(r, indent=2))
        m = r["methods"]
        print(f"{name:7s} " + " | ".join(
            f"{k.split(' ')[0]}{'-m' if 'matched' in k else ''}{'-avg' if 'avg' in k else ''}: conc {v['concentration_top5pct_rows']:.2f} "
            f"lev×{v['profile']['median_leverage_ratio']:.1f} ρlev {v['spearman_with_leverage']:.2f} ρ|m| {v['spearman_with_abs_margin']:.2f}"
            for k, v in m.items()) + f" | J(sgd,ra) {r['overlap']['jaccard_sgd_textbook_vs_ragmm']:.2f} [{r['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
