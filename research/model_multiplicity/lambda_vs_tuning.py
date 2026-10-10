"""
Experiment 1b (exploratory): reduce seed-dependence by changing the objective
(larger L2 penalty lambda) or by tuning the optimizer? See PLAN.md section 7.

Run from the repository root:
    python research/model_multiplicity/lambda_vs_tuning.py
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loan_experiment import (COST_BAD_APPROVED, COST_GOOD_REJECTED, COST_THRESHOLD,  # noqa: E402
                             auc, batch_indices, constants, load_data, run_gmm, sigmoid,
                             solve_exact, tail_stats, tune_for_mean)
from paper_tuning import ra_gmm_params  # noqa: E402


def quality(W, Xte, yte, thr):
    P = sigmoid(W @ Xte.T)
    approve = P >= thr
    good = yte > 0
    cost = (COST_BAD_APPROVED * (approve & ~good) + COST_GOOD_REJECTED * (~approve & good)).mean(1)
    aucs = np.array([auc(P[s], yte) for s in range(len(W))])
    return approve, cost, aucs


def summarize(W, Xte, yte, thr, ref_decisions, own_decisions):
    approve, cost, aucs = quality(W, Xte, yte, thr)
    F_ref = (approve != ref_decisions).sum(1)
    F_own = (approve != own_decisions).sum(1)
    unanimous = approve.all(0) | (~approve).all(0)
    return dict(
        flips_vs_reference=tail_stats(F_ref),
        flips_vs_own_ideal=tail_stats(F_own),
        ambiguity=float(1 - unanimous.mean()),
        cost_per_applicant=tail_stats(cost),
        auc_mean=float(aucs.mean()),
        approval_rate=float(approve.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam0", type=float, default=0.01)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    t0 = time.time()

    Xtr, ytr, Xte, yte = load_data()
    n, d = Xtr.shape
    thr = COST_THRESHOLD
    idx = batch_indices(range(args.seeds), args.K, n, args.batch)

    w0, _ = solve_exact(Xtr, ytr, args.lam0)
    ref = sigmoid(Xte @ w0) >= thr                      # decisions of the intended model
    _, ref_cost, ref_auc = quality(w0[None], Xte, yte, thr)

    alphas_mult = np.logspace(-2, math.log10(2.0), 12)
    rows = []

    def add(arm, lam, name, cfg, wstar, fstar, L):
        W, _ = run_gmm(Xtr, ytr, lam, idx, **cfg)
        own = sigmoid(Xte @ wstar) >= thr
        r = summarize(W, Xte, yte, thr, ref, own)
        _, ic, ia = quality(wstar[None], Xte, yte, thr)
        r.update(arm=arm, lam=lam, method=name,
                 params={k: (float(v) if not isinstance(v, bool) else v) for k, v in cfg.items()},
                 bias_flips=int((own != ref).sum()),
                 ideal_cost=float(ic[0]), ideal_auc=float(ia[0]))
        rows.append(r)
        print(f"{arm:5s} lam={lam:<5} {name:30s} flips_ref {r['flips_vs_reference']['mean']:6.1f} "
              f"(p95 {r['flips_vs_reference']['p95']:5.1f}) bias {r['bias_flips']:3d} "
              f"cost {r['cost_per_applicant']['mean']:.3f} auc {r['auc_mean']:.3f}  [{time.time()-t0:.0f}s]")

    for lam in [args.lam0] + args.lams:
        L, mu = constants(Xtr, lam)
        wstar, fstar = solve_exact(Xtr, ytr, lam)
        kappa = L / mu
        m = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
        base = lam == args.lam0
        arm_default = "ref" if base else "B"
        add(arm_default, lam, "SGD (textbook)", dict(alpha=1 / L, beta=0.0, gamma=0.0), wstar, fstar, L)
        add(arm_default, lam, "Nesterov (textbook)", dict(alpha=1 / L, beta=m, gamma=m), wstar, fstar, L)
        avg = tune_for_mean(Xtr, ytr, lam, fstar, args.K, args.batch, L,
                            [dict(alpha=c / L, beta=0.0, gamma=0.0, tail_average=True) for c in alphas_mult])
        add("A" if base else "A+B", lam, "SGD + tail averaging (tuned)", avg, wstar, fstar, L)
        if base:
            sgd = tune_for_mean(Xtr, ytr, lam, fstar, args.K, args.batch, L,
                                [dict(alpha=c / L, beta=0.0, gamma=0.0) for c in alphas_mult])
            add("A", lam, "SGD (tuned for mean)", sgd, wstar, fstar, L)
            p = ra_gmm_params(L, mu, d, args.zeta, args.eps)
            add("A", lam, f"RA-GMM (eps={args.eps})", dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]),
                wstar, fstar, L)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lambda_vs_tuning.json").write_text(json.dumps(dict(
        config=vars(args), reference=dict(lam=args.lam0, approval_rate=float(ref.mean()),
                                          cost=float(ref_cost[0]), auc=float(ref_auc[0])),
        rows=rows), indent=2))
    print(f"wrote {out/'lambda_vs_tuning.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
