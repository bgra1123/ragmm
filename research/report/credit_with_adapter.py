"""
Credit outcomes with and without the polish step. Plan: PLAN.md, Addendum 2.
Run from the repository root:  python research/report/credit_with_adapter.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for sub in ("polish", "linesearch", "optimizers", "model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parents[1]))
import loan_experiment as le  # noqa: E402
import linesearch_experiment as ls  # noqa: E402
import optimizer_experiments as oe  # noqa: E402
import polish_experiments as pe  # noqa: E402
import run_gate0 as g0  # noqa: E402
from risk_averse_tuning import RAGMMBounds  # noqa: E402

LAM, BATCH, K1, SEEDS = 0.01, 16, 2000, 300
THRESHOLDS = (0.5, 5 / 6)


def outcomes(W, Xte, good, pstar, thr_list=THRESHOLDS):
    P = le.sigmoid(W @ Xte.T)
    res = {}
    for thr in thr_list:
        appr = P >= thr
        ex = pstar >= thr
        wd, wa = (~appr) & good[None], appr & (~good)[None]
        ind = ((~appr) & good[None] & ex[None]).sum(1) + (appr & (~good)[None] & (~ex)[None]).sum(1)
        share = appr.mean(0)
        res[f"{thr:.3f}"] = dict(
            wrongly_denied_n=float(wd.sum(1).mean()), wrongly_denied_rate=float(wd.sum(1).mean() / good.sum()),
            wrongly_approved_n=float(wa.sum(1).mean()), wrongly_approved_rate=float(wa.sum(1).mean() / (~good).sum()),
            default_among_approved=float((wa.sum(1) / np.maximum(appr.sum(1), 1)).mean()),
            cost=float((5.0 * wa.sum(1) + wd.sum(1)).mean() / len(good)),
            changed=float((appr != ex[None]).sum(1).mean()), coin_flip=int((np.minimum(share, 1 - share) >= 0.10).sum()),
            induced_errors=float(ind.mean()))
    return res, (5.0 * ((P >= 0.5) & (~good)[None]).sum(1) + ((P < 0.5) & good[None]).sum(1))


def boot(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = [a[j].mean() - b[j].mean() for j in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return dict(diff=float(a.mean() - b.mean()), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])


def main():
    X, y, Xte, yte = g0.prepare("german")
    n, d = X.shape
    L, mu = le.constants(X, LAM)
    wstar, fstar = le.solve_exact(X, y, LAM)
    pstar = le.sigmoid(Xte @ wstar)
    good = yte > 0
    idx1 = le.batch_indices(range(SEEDS), K1, n, BATCH)
    ra = RAGMMBounds(zeta=0.05).solve(L, mu, d, rate_slack=0.25)
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    careful = le.tune_for_mean(X, y, LAM, fstar, K1, BATCH, L, [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    stage1 = {
        "SGD (textbook)": le.run_gmm(X, y, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0],
        "SGD + momentum 0.9 (lr 0.01)": oe.run_opt("SGD+momentum", X, y, idx1, 0.01),
        "Adam (lr 0.001, default)": oe.run_opt("Adam", X, y, idx1, 0.001),
        "Adam (lr 0.01)": oe.run_opt("Adam", X, y, idx1, 0.01),
        "ADMM (5 iterations)": np.stack([pe.admm(X, y, np.random.default_rng(s)) for s in range(SEEDS)]),
        "RA-GMM from scratch (paper)": le.run_gmm(X, y, LAM, idx1, alpha=ra.alpha, beta=ra.beta, gamma=ra.gamma)[0],
        "SGD careful (tuned + averaging)": le.run_gmm(X, y, LAM, idx1, **careful)[0],
    }
    exact_out, _ = outcomes(wstar[None].repeat(2, 0), Xte, good, pstar)
    out = dict(exact_model=exact_out, n_good=int(good.sum()), n_bad=int((~good).sum()), seeds=SEEDS, trainers={})
    rng = np.random.default_rng(0)
    for nm, W1 in stage1.items():
        t = dict(none=outcomes(W1, Xte, good, pstar)[0], polished={})
        _, cost_none = outcomes(W1, Xte, good, pstar)
        wd_none = ((le.sigmoid(W1 @ Xte.T) < 0.5) & good[None]).sum(1)
        for B in (5, 20):
            K2 = round(B * n / BATCH)
            idx2 = le.batch_indices(range(100_000, 100_000 + SEEDS), K2, n, BATCH)
            sel, _ = pe.select_polish(X, y, W1, K2, rng)
            W_ad = pe.run_from(W1, X, y, idx2, **sel)
            W_ls, _ = ls.gd(W1, X, y, B, "LS", 0, L)
            W_r1, _ = ls.gd(W1, X, y, B, "R1", 5.0, L)
            for pn, W in (("adapter", W_ad), ("GD exact line search", W_ls), ("risk-averse step R1 (θ=5)", W_r1)):
                o, cost = outcomes(W, Xte, good, pstar)
                wd = ((le.sigmoid(W @ Xte.T) < 0.5) & good[None]).sum(1)
                o["paired_vs_none"] = dict(cost_half=boot(cost, cost_none), wrongly_denied_half=boot(wd, wd_none))
                t["polished"].setdefault(str(B), {})[pn] = o
        out["trainers"][nm] = t
        print(nm, {B: {p: round(v["0.833"]["cost"], 3) for p, v in c.items()} for B, c in t["polished"].items()},
              "none", round(t["none"]["0.833"]["cost"], 3), flush=True)
    (HERE / "credit_with_adapter.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
