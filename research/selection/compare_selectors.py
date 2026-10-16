"""
Compare parameter selectors for the polish stage. Design: PLAN.md (pre-registered).
Run from the repository root:  python research/selection/compare_selectors.py german compas pima heart
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
from risk_averse_tuning.quadratic import evar_quadratic, gmm_matrices  # noqa: E402
from risk_averse_tuning.tail_report import empirical_evar  # noqa: E402

LAM, BATCH, K1, K2, ZETA = 0.01, 16, 2000, 500, 0.05
MULTS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def candidates(L):
    out = []
    for c in MULTS:
        out.append((c / L, 0.0, 0.0))
        for b in (0.5, 0.9, 0.98):
            out += [(c / L, b, 0.0), (c / L, b, b)]
    return out


def stage1(X, y, seeds, L):
    idx = le.batch_indices(seeds, K1, X.shape[0], BATCH)
    return le.run_gmm(X, y, LAM, idx, alpha=1 / L, beta=0.0, gamma=0.0)[0]


def select_B(H, Sig, e0s, cands):
    vals = [np.mean([evar_quadratic(H, a, b, g, K2, Sig, e0, ZETA) for e0 in e0s]) for a, b, g in cands]
    vals = np.where(np.isfinite(vals), vals, np.inf)
    return int(np.argmin(vals))


def select_C(H, Gc, e0s, cands, rng, paths=40):
    d = H.shape[0]
    n = len(Gc)
    idxs = rng.integers(0, n, size=(K2, len(e0s) * paths, BATCH))          # same draws for every candidate
    noise = np.stack([Gc[idxs[k]].mean(1) for k in range(K2)])             # (K2, paths_total, d)
    E0 = np.repeat(np.array(e0s), paths, axis=0)
    vals = []
    for a, b, g in cands:
        A, Bm = gmm_matrices(H, a, b, g)
        if max(abs(np.linalg.eigvals(A))) >= 1.0 + 1e-9:
            vals.append(np.inf)
            continue
        s = np.hstack([E0, E0])
        At, Bt = A.T, Bm[:d, :d].T
        for k in range(K2):
            s = s @ At
            s[:, :d] += noise[k] @ Bt * 1.0
        e = s[:, :d]
        vals.append(empirical_evar(0.5 * np.einsum("ni,ij,nj->n", e, H, e), ZETA))
    return int(np.argmin(vals))


def select_A(X, y, W1p, cands, reps=20):
    vals = []
    W0 = np.repeat(W1p, reps, axis=0)
    idx = le.batch_indices(range(500_000, 500_000 + len(W0)), K2, X.shape[0], BATCH)
    for a, b, g in cands:
        W = pe.run_from(W0, X, y, idx, alpha=a, beta=b, gamma=g)
        f = le.objective(W, X, y, LAM)
        vals.append(empirical_evar(f - f.min() + 1e-12, ZETA) + f.min() - 1e-12 if np.all(np.isfinite(f)) else np.inf)
    return int(np.argmin(vals))


def evaluate_all(X, y, Xte, yte, wstar, fstar, W1, cands):
    idx = le.batch_indices(range(900_000, 900_000 + len(W1)), K2, X.shape[0], BATCH)
    out = []
    for a, b, g in cands:
        W = pe.run_from(W1, X, y, idx, alpha=a, beta=b, gamma=g)
        S = le.objective(W, X, y, LAM) - fstar
        if not np.all(np.isfinite(S)):
            out.append(dict(evar=math.inf, cvar5=math.inf, mean=math.inf, changed=math.inf))
            continue
        q95 = np.quantile(S, 0.95)
        ch = ((le.sigmoid(W @ Xte.T) >= 0.5) != (le.sigmoid(Xte @ wstar) >= 0.5)[None]).sum(1)
        out.append(dict(evar=empirical_evar(S, ZETA), cvar5=float(S[S >= q95].mean()), mean=float(S.mean()),
                        changed=float(ch.mean())))
    return out


def run(name):
    X, y, Xte, yte = g0.prepare(name)
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    cands = candidates(L)
    W_eval1 = stage1(X, y, range(300), L)
    t0 = time.time()
    ev = evaluate_all(X, y, Xte, yte, wstar, fstar, W_eval1, cands)
    best = min(range(len(cands)), key=lambda i: ev[i]["evar"])
    res = dict(dataset=name, n_candidates=len(cands), oracle_best=dict(index=best, params=cands[best], **ev[best]),
               eval_seconds=time.time() - t0, repeats=[])
    rng = np.random.default_rng(0)
    for rep in range(5):
        seeds = range(1000 + 1000 * rep, 1000 + 1000 * rep + 12)
        W1p = stage1(X, y, seeds, L)
        wbar = W1p.mean(0)
        H, Sig = pe.hessian(X, y, wbar), pe.grad_cov(X, y, wbar)
        G = -(y * le.sigmoid(-y * (X @ wbar)))[:, None] * X
        Gc = G - G.mean(0)
        e0s = [-np.linalg.solve(pe.hessian(X, y, w), -(X.T @ (y * le.sigmoid(-y * (X @ w)))) / n + LAM * w) for w in W1p]
        sel, secs = {}, {}
        for nm, fn in (("B", lambda: select_B(H, Sig, e0s, cands)), ("C", lambda: select_C(H, Gc, e0s, cands, rng)),
                       ("A", lambda: select_A(X, y, W1p, cands))):
            t = time.time()
            i = fn()
            secs[nm] = time.time() - t
            sel[nm] = dict(index=i, params=cands[i], regret=ev[i]["evar"] / ev[best]["evar"], cvar5=ev[i]["cvar5"],
                           mean=ev[i]["mean"], changed=ev[i]["changed"], seconds=secs[nm])
        res["repeats"].append(sel)
    res["none"] = None
    print(name, "oracle", np.round(np.array(cands[best]) * [L, 1, 1], 3).tolist(), "evar", round(ev[best]["evar"], 5),
          {k: [round(r[k]["regret"], 2) for r in res["repeats"]] for k in "BCA"},
          {k: round(float(np.mean([r[k]["seconds"] for r in res["repeats"]])), 1) for k in "BCA"}, flush=True)
    return res


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["german", "compas", "pima", "heart"]):
        (HERE / "results" / f"{name}.json").write_text(json.dumps(run(name), indent=2, default=float))
