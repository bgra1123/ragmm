"""Pre-registered test (PLAN.md): do rare rows flip more because the boundary moves more where data is thin?
Run from the repository root:  python research/rare_flip/rare_flip.py german compas adult bank pima heart"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for sub in ("linesearch", "polish", "optimizers", "model_multiplicity", "gate0", "groups", "risk_effect"):
    sys.path.insert(0, str(HERE.parent / sub))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
import stepmatch_experiment as sm  # noqa: E402
from group_experiment import test_mask  # noqa: E402
from risk_averse_tuning.tail_report import leverage  # noqa: E402

LAM, BATCH, K, SEEDS = 0.01, 16, 2000, 200
sig = le.sigmoid


def raw_frames(name):
    """Raw held-out rows (for categorical rarity) using the same split as g0.prepare."""
    if name == "german":
        df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
        yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
        m = test_mask(yall, 0)
        return df.drop(columns=[le.TARGET])[~m], df.drop(columns=[le.TARGET])[m]
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
    return Xdf[~mask], Xdf[mask]


def cat_rarity(tr, te):
    cols = [c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])]
    if not cols:
        return None
    out = np.ones(len(te))
    for c in cols:
        freq = tr[c].astype(str).value_counts(normalize=True)
        out = np.minimum(out, te[c].astype(str).map(freq).fillna(0.0).to_numpy())
    return out


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def partial_spearman(y, x, z):
    ry, rx, rz = (pd.Series(v).rank().to_numpy() for v in (y, x, z))
    def resid(a):
        A = np.column_stack([rz, np.ones_like(rz)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(resid(ry), resid(rx))[0, 1])


def decile_test(flip, rarity, margin, rng, n_boot=1000):
    dec = pd.qcut(pd.Series(margin).rank(method="first"), 10, labels=False).to_numpy()
    rare = rarity >= np.quantile(rarity, 0.9)
    wins, rows = 0, []
    for d in range(10):
        m = dec == d
        a, b = flip[m & rare], flip[m & ~rare]
        if len(a) and len(b):
            rows.append((d, float(a.mean()), float(b.mean()), int(len(a))))
            wins += a.mean() > b.mean()
    def pooled(idx):
        num = den = 0.0
        for d in range(10):
            m = (dec == d)
            sel = np.zeros_like(m); sel[idx] = True
            a, b = flip[m & rare & sel], flip[m & ~rare & sel]
            if len(a) and len(b):
                w = m[sel].sum()
                num += w * a.mean(); den += w * b.mean()
        return num / den if den > 0 else np.nan
    n = len(flip)
    ratio = pooled(np.arange(n))
    boots = [pooled(rng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return dict(wins=int(wins), deciles=rows, ratio=float(ratio), ci=[float(lo), float(hi)],
                rare_flip=float(flip[rare].mean()), rest_flip=float(flip[~rare].mean()))


def run(name):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    tr_raw, te_raw = raw_frames(name)
    assert len(te_raw) == len(yte), (len(te_raw), len(yte))
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    margin = np.abs(Xte @ wstar)
    ex = Xte @ wstar >= 0
    lev = leverage(X, y, wstar, LAM, Xte)
    nn = NearestNeighbors(n_neighbors=10).fit(X[:, :-1])
    knn = nn.kneighbors(Xte[:, :-1], return_distance=True)[0].mean(1)
    cat = cat_rarity(tr_raw, te_raw)
    idx = le.batch_indices(range(SEEDS), K, n, BATCH)
    rng = np.random.default_rng(0)
    out = dict(dataset=name, n_train=n, n_test=len(yte), d=d, seeds=SEEDS, regimes={})
    W_fast = le.run_gmm(X, y, LAM, idx, alpha=1 / L, beta=0.0, gamma=0.0)[0]
    W_slow = le.run_gmm(X, y, LAM, idx, alpha=0.01, beta=0.0, gamma=0.0)[0]
    W_pol = sm.trajectory(W_fast, X, y, 20, "LS")[0][20]
    for reg, W in (("step 1/L", W_fast), ("lr 0.01", W_slow), ("step 1/L + 20 line-search steps", W_pol)):
        S = W @ Xte.T
        flip = ((S >= 0) != ex[None]).mean(0)
        std = S.std(0)
        r = dict(flip_rate=float(flip.mean()), disagreement=float(np.mean(2 * (S >= 0).mean(0) * (1 - (S >= 0).mean(0))) * SEEDS / (SEEDS - 1)),
                 spearman_std_leverage=spearman(std, lev), spearman_std_knn=spearman(std, knn),
                 spearman_flip_knn=spearman(flip, knn), spearman_flip_margin=spearman(flip, margin),
                 partial_spearman_flip_knn_given_margin=partial_spearman(flip, knn, margin),
                 partial_spearman_flip_leverage_given_margin=partial_spearman(flip, lev, margin),
                 knn=decile_test(flip, knn, margin, rng), leverage=decile_test(flip, lev, margin, rng))
        if cat is not None:
            r["spearman_std_catfreq"] = spearman(std, cat)
            r["catfreq"] = decile_test(flip, -cat, margin, rng)   # rare = lowest frequency
        out["regimes"][reg] = r
        print(name, reg, f"flip {flip.mean():.3f} rho(std,lev) {r['spearman_std_leverage']:.2f} rho(std,knn) {r['spearman_std_knn']:.2f} "
              f"knn ratio {r['knn']['ratio']:.2f} wins {r['knn']['wins']}/10 [{time.time() - t0:.0f}s]", flush=True)
    # correlation among predictors, for the record
    out["predictor_spearman"] = dict(knn_margin=spearman(knn, margin), leverage_margin=spearman(lev, margin), knn_leverage=spearman(knn, lev))
    return out


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ["german", "compas", "adult", "bank", "pima", "heart"]):
        (HERE / "results" / f"{name}.json").write_text(json.dumps(run(name), indent=2))
