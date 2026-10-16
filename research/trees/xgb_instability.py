"""
Seed instability of XGBoost and levers that reduce it. Design: PLAN.md (pre-registered).
Run from the repository root:  python research/trees/xgb_instability.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import xgboost as xgb

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402

SEEDS = 200
LAM = 0.01


def logloss(p, y01):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y01 * np.log(p) + (1 - y01) * np.log(1 - p)).mean(-1)


def fit_xgb(X, y01, seed, **kw):
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    params.update(kw)
    return xgb.XGBClassifier(**params).fit(X, y01)


def newton_logistic(F, y, lam):
    """Minimizer of the L2 logistic objective on leaf indicators (thousands of columns: L-BFGS, gradient norm < 1e-7)."""
    from scipy.optimize import minimize
    n = len(y)

    def fg(w):
        m = y * (F @ w)
        f = np.logaddexp(0.0, -m).mean() + 0.5 * lam * w @ w
        g = -(F.T @ (y * le.sigmoid(-m))) / n + lam * w
        return f, g

    r = minimize(fg, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-7, ftol=1e-15))
    return r.x


def leaf_features(models_leaves, enc=None):
    """One-hot encode leaf indices per tree. Returns (matrix, encoder dict)."""
    L = models_leaves
    if enc is None:
        enc = [np.unique(L[:, t]) for t in range(L.shape[1])]
    cols = []
    for t, u in enumerate(enc):
        cols.append((L[:, t][:, None] == u[None, :]).astype(float))
    return np.hstack(cols), enc


def disagreement(P, thr=0.5):
    dec = P >= thr
    p = dec.mean(0)
    S = len(P)
    return dict(retrain_disagreement=float(np.mean(2 * p * (1 - p)) * S / (S - 1)),
                coin_flip_rows=int((np.minimum(p, 1 - p) >= 0.10).sum()), ever_changed_rows=int(((p > 0) & (p < 1)).sum()))


def summarize(P, yte01, consensus_dec, n_units):
    dec = P >= 0.5
    ll = logloss(P, yte01[None, :])
    q95 = np.quantile(ll, 0.95)
    k5 = max(1, len(P) // 20)
    aucs = np.array([le.auc(P[s], np.where(yte01 > 0, 1.0, -1.0)) for s in range(len(P))])
    out = dict(units=int(n_units), **disagreement(P), logloss_mean=float(ll.mean()), logloss_p95=float(q95),
               logloss_cvar5=float(np.sort(ll)[-k5:].mean()), auc_mean=float(aucs.mean()),
               auc_worst5=float(np.sort(aucs)[:k5].mean()),
               changed_vs_consensus=float((dec != consensus_dec[None]).sum(1).mean()))
    return out


def run(name):
    t0 = time.time()
    Xtr, ytr, Xte, yte = g0.prepare(name)
    Xtr_f, Xte_f = Xtr[:, :-1], Xte[:, :-1]                   # drop the intercept column for trees
    y01tr, y01te = (ytr > 0).astype(int), (yte > 0).astype(int)
    res = dict(dataset=name, xgboost=xgb.__version__, n_train=len(ytr), n_test=len(yte), configs={})

    def sample(**kw):
        return np.stack([fit_xgb(Xtr_f, y01tr, s, **kw).predict_proba(Xte_f)[:, 1] for s in range(SEEDS)])

    P_def = sample()
    consensus = P_def.mean(0) >= 0.5
    res["configs"]["default"] = summarize(P_def, y01te, consensus, SEEDS)
    P_ns = np.stack([fit_xgb(Xtr_f, y01tr, s, subsample=1.0, colsample_bytree=1.0).predict_proba(Xte_f)[:, 1] for s in range(10)])
    res["configs"]["no subsampling"] = dict(summarize(P_ns, y01te, consensus, 10),
                                            identical_predictions=bool(np.all(P_ns == P_ns[0])))
    P_slow = sample(learning_rate=0.03, n_estimators=1000)
    res["configs"]["slow (lr 0.03, 1000 trees)"] = summarize(P_slow, y01te, consensus, SEEDS)
    for m in (5, 20):
        Pm = P_def.reshape(SEEDS // m, m, -1).mean(1)
        res["configs"][f"seed-average m={m}"] = summarize(Pm, y01te, consensus, SEEDS // m)
    # last-layer refit on leaf indicators
    P_leaf = []
    for s in range(SEEDS):
        mdl = fit_xgb(Xtr_f, y01tr, s)
        Ltr, Lte = mdl.apply(Xtr_f), mdl.apply(Xte_f)
        Ftr, enc = leaf_features(Ltr)
        Fte, _ = leaf_features(Lte, enc)
        Ftr, Fte = np.hstack([Ftr, np.ones((len(Ftr), 1))]), np.hstack([Fte, np.ones((len(Fte), 1))])
        w = newton_logistic(Ftr, ytr, LAM)
        P_leaf.append(le.sigmoid(Fte @ w))
    P_leaf = np.stack(P_leaf)
    res["configs"]["last-layer refit (leaf one-hot, L2 logistic)"] = summarize(P_leaf, y01te, consensus, SEEDS)
    res["configs"]["leaf refit + seed-average m=20"] = summarize(P_leaf.reshape(SEEDS // 20, 20, -1).mean(1), y01te, consensus, SEEDS // 20)
    # logistic baselines (Gate 0 settings)
    L, mu = le.constants(Xtr, LAM)
    wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
    idx = le.batch_indices(range(SEEDS), 2000, len(ytr), 16)
    base = {"logistic SGD lr=0.01": dict(alpha=0.01, beta=0.0, gamma=0.0)}
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    base["logistic careful SGD (tuned + averaging)"] = le.tune_for_mean(
        Xtr, ytr, LAM, fstar, 2000, 16, L, [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    for nm, cfg in base.items():
        W, _ = le.run_gmm(Xtr, ytr, LAM, idx, **cfg)
        res["configs"][nm] = summarize(le.sigmoid(W @ Xte.T), y01te, consensus, SEEDS)
    res["logistic_exact_auc"] = float(le.auc(le.sigmoid(Xte @ wstar), yte))
    res["seconds"] = time.time() - t0
    return res


def main():
    (HERE / "results").mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        r = run(name)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(r, indent=2))
        print(name, f"[{r['seconds']:.0f}s]", {k: (round(v['retrain_disagreement'], 4), round(v['auc_mean'], 3)) for k, v in r["configs"].items()}, flush=True)


if __name__ == "__main__":
    main()
