"""XGBoost and MLP: seed instability and credit outcomes. Plan: PLAN.md. Run: python research/sota/sota_experiment.py german compas"""
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402

SEEDS, LAM = 200, 0.01


def head_fit(F, y):
    n = len(y)
    F = np.hstack([F, np.ones((n, 1))])

    def fg(w):
        m = y * (F @ w)
        return np.logaddexp(0, -m).mean() + 0.5 * LAM * w @ w, -(F.T @ (y * le.sigmoid(-m))) / n + LAM * w

    return minimize(fg, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-8, ftol=1e-15)).x


def with_bias(F):
    return np.hstack([F, np.ones((len(F), 1))])


def acts(mlp, X):
    h = X
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0)
    return h


def metrics(P, yte, cons, with_credit):
    S = len(P)
    out = {}
    dec = P >= 0.5
    p = dec.mean(0)
    aucs = np.array([le.auc(P[s], yte) for s in range(S)])
    out.update(units=int(S), retrain_disagreement=float(np.mean(2 * p * (1 - p)) * S / (S - 1)) if S > 1 else None,
               coin_flip_rows=int((np.minimum(p, 1 - p) >= 0.10).sum()), auc_mean=float(aucs.mean()),
               changed_vs_consensus=float((dec != cons[None]).sum(1).mean()))
    if with_credit:
        good = yte > 0
        for thr in (0.5, 5 / 6):
            a = P >= thr
            wd, wa = (~a) & good[None], a & (~good)[None]
            cost = (5.0 * wa.sum(1) + wd.sum(1)) / len(yte)
            out[f"{thr:.3f}"] = dict(wrongly_denied_rate=float(wd.sum(1).mean() / good.sum()),
                                     wrongly_approved_rate=float(wa.sum(1).mean() / (~good).sum()),
                                     default_among_approved=float((wa.sum(1) / np.maximum(a.sum(1), 1)).mean()),
                                     cost_mean=float(cost.mean()), cost_p05=float(np.quantile(cost, .05)),
                                     cost_p95=float(np.quantile(cost, .95)))
    return out


def avg(P, m):
    return P.reshape(len(P) // m, m, -1).mean(1)


def run(name):
    t0 = time.time()
    X, y, Xte, yte = g0.prepare(name)
    Xf, Xtef = X[:, :-1], Xte[:, :-1]
    y01 = (y > 0).astype(int)
    credit = name == "german"
    res = dict(dataset=name, n_train=len(y), n_test=len(yte), configs={})
    # XGBoost
    def fx(seed, **kw):
        p = dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
        p.update(kw)
        return xgb.XGBClassifier(**p).fit(Xf, y01)
    mods = [fx(s) for s in range(SEEDS)]
    Px = np.stack([m.predict_proba(Xtef)[:, 1] for m in mods])
    cons_x = Px.mean(0) >= 0.5
    cfg = {"XGBoost default": Px, "XGBoost seed-average m=5": avg(Px, 5), "XGBoost seed-average m=20": avg(Px, 20)}
    Pn = np.stack([fx(s, subsample=1.0, colsample_bytree=1.0).predict_proba(Xtef)[:, 1] for s in range(5)])
    cfg["XGBoost no subsampling"] = Pn
    ref = {}
    for m in mods:
        Ltr, Lte = m.apply(Xf), m.apply(Xtef)
        enc = [np.unique(Ltr[:, t]) for t in range(Ltr.shape[1])]
        F = lambda L: np.hstack([(L[:, t][:, None] == u[None, :]).astype(float) for t, u in enumerate(enc)])
        w = head_fit(F(Ltr), y)
        ref.setdefault("p", []).append(le.sigmoid(with_bias(F(Lte)) @ w))
    Pxr = np.stack(ref["p"])
    cfg["XGBoost + exact head refit"] = Pxr
    cfg_cons = {k: cons_x for k in cfg}
    # MLP
    Pm, Pmr = [], []
    for s in range(SEEDS):
        mlp = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=1e-3, batch_size=32,
                            max_iter=60, alpha=1e-3, random_state=s, early_stopping=False, tol=0, n_iter_no_change=1000).fit(Xf, y01)
        Pm.append(mlp.predict_proba(Xtef)[:, 1])
        w = head_fit(acts(mlp, Xf), y)
        Pmr.append(le.sigmoid(with_bias(acts(mlp, Xtef)) @ w))
    Pm, Pmr = np.stack(Pm), np.stack(Pmr)
    cons_m = Pm.mean(0) >= 0.5
    for k, v in {"MLP default": Pm, "MLP seed-average m=5": avg(Pm, 5), "MLP seed-average m=20": avg(Pm, 20),
                 "MLP + exact head refit": Pmr, "MLP + head refit + seed-average m=5": avg(Pmr, 5),
                 "MLP + head refit + seed-average m=20": avg(Pmr, 20)}.items():
        cfg[k] = v
        cfg_cons[k] = cons_m
    # logistic references
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    cfg["Logistic regression (exact)"] = np.repeat(le.sigmoid(Xte @ wstar)[None], 2, 0)
    cfg_cons["Logistic regression (exact)"] = le.sigmoid(Xte @ wstar) >= 0.5
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    careful = le.tune_for_mean(X, y, LAM, fstar, 2000, 16, L, [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    W, _ = le.run_gmm(X, y, LAM, le.batch_indices(range(SEEDS), 2000, len(y), 16), **careful)
    cfg["Logistic careful SGD"] = le.sigmoid(W @ Xte.T)
    cfg_cons["Logistic careful SGD"] = le.sigmoid(Xte @ wstar) >= 0.5
    for k, P in cfg.items():
        res["configs"][k] = metrics(P, yte, cfg_cons[k], credit)
    res["seconds"] = time.time() - t0
    print(name, f"[{res['seconds']:.0f}s]", {k: (round(v['retrain_disagreement'], 3) if v['retrain_disagreement'] is not None else None, round(v['auc_mean'], 3)) for k, v in res['configs'].items()}, flush=True)
    return res


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["german", "compas"]):
        (HERE / "results" / f"{name}.json").write_text(json.dumps(run(name), indent=2))
