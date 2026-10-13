"""
Wrongly denied vs wrongly approved loans across retrains. Plan: PLAN.md (addendum).
Run from the repository root:  python research/report/error_types.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
sys.path.insert(0, str(HERE.parent / "gate0"))
sys.path.insert(0, str(HERE.parent / "tail_analysis"))
import loan_experiment as le  # noqa: E402
import run_gate0 as g0  # noqa: E402
import tail_rows as tr  # noqa: E402
from paper_tuning import ra_gmm_params  # noqa: E402

LAM, BATCH, K, SEEDS, ZETA = 0.01, 16, 2000, 1000, 0.05
COLS = ["status_of_existing_checking_account", "credit_history", "purpose", "savings_account_and_bonds",
        "present_employment_since", "other_debtors_or_guarantors", "property", "job", "housing",
        "other_installment_plans", "telephone", "duration_in_month", "credit_amount"]


def features(raw):
    D = pd.get_dummies(raw[COLS[:-2]], dtype=float)
    D["duration > 24 months"] = (raw["duration_in_month"] > 24).astype(float)
    D["credit amount > 4,000 DM"] = (raw["credit_amount"] > 4000).astype(float)
    return D


def characteristics(D, counts, base_mask, min_cov=0.25, top=4):
    """counts: error events per applicant. Coverage = share of events with the feature; base = share of the
    reference class (base_mask applicants) with it."""
    tot = counts.sum()
    rows = []
    for c in D.columns:
        has = D[c].to_numpy() > 0
        cov = counts[has].sum() / tot
        base = has[base_mask].mean()
        if cov >= min_cov and base > 0:
            rows.append(dict(feature=c, coverage=float(cov), base=float(base), lift=float(cov / base)))
    return sorted(rows, key=lambda r: -r["lift"])[:top]


def main():
    Xtr, ytr, Xte, yte = g0.prepare("german")
    Xdf, yall, train_mask = tr.raw_frame("german")
    raw_te = Xdf[~train_mask].reset_index(drop=True)
    D = features(raw_te)
    n, d = Xtr.shape
    L, mu = le.constants(Xtr, LAM)
    wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
    p_star = le.sigmoid(Xte @ wstar)
    good = yte > 0
    p = ra_gmm_params(L, mu, d, ZETA, 0.25)
    ra = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
    idx_t = le.batch_indices(range(10_000, 10_100), K, n, BATCH)
    Wt, _ = le.run_gmm(Xtr, ytr, LAM, idx_t, **ra)
    a_match, _ = tr.matched_alpha(Xtr, ytr, fstar, float(np.mean(le.objective(Wt, Xtr, ytr, LAM) - fstar)), L)
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    careful = le.tune_for_mean(Xtr, ytr, LAM, fstar, K, BATCH, L,
                               [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    methods = {"SGD (textbook)": dict(alpha=1 / L, beta=0.0, gamma=0.0), "RA-GMM": ra,
               "SGD (same average as RA-GMM)": dict(alpha=a_match, beta=0.0, gamma=0.0),
               "SGD (careful: tuned + averaging)": careful}
    idx = le.batch_indices(range(SEEDS), K, n, BATCH)
    out = dict(n_good=int(good.sum()), n_bad=int((~good).sum()), methods={})
    per_model = {}
    for nm, cfg in methods.items():
        W, _ = le.run_gmm(Xtr, ytr, LAM, idx, **cfg)
        P = le.sigmoid(W @ Xte.T)
        res = {}
        for thr in (0.5, 5 / 6):
            appr = P >= thr
            ex_appr = p_star >= thr
            wd = (~appr) & good[None]                       # wrongly denied events (models x applicants)
            wa = appr & (~good)[None]                       # wrongly approved
            wd_ind = wd & ex_appr[None]                     # training-induced: exact model approves this good applicant
            wa_ind = wa & (~ex_appr)[None]                  # exact model denies this bad applicant
            r = dict(
                wrongly_denied_rate=float(wd.sum(1).mean() / good.sum()), wrongly_denied_n=float(wd.sum(1).mean()),
                wrongly_approved_rate=float(wa.sum(1).mean() / (~good).sum()), wrongly_approved_n=float(wa.sum(1).mean()),
                default_rate_among_approved=float((wa.sum(1) / np.maximum(appr.sum(1), 1)).mean()),
                exact_wrongly_denied_n=int(((~ex_appr) & good).sum()), exact_wrongly_approved_n=int((ex_appr & ~good).sum()),
                induced_wrongly_denied_n=float(wd_ind.sum(1).mean()), induced_wrongly_approved_n=float(wa_ind.sum(1).mean()),
                cost_per_applicant=float((5.0 * wa.sum(1) + 1.0 * wd.sum(1)).mean() / len(yte)))
            r["_wd"], r["_wa"], r["_wd_ind"], r["_wa_ind"] = wd, wa, wd_ind, wa_ind
            res[f"{thr:.3f}"] = r
        per_model[nm] = res
        out["methods"][nm] = {t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in res.items()}
    # characteristics (threshold 0.5), textbook SGD and RA-GMM
    for nm in ("SGD (textbook)", "RA-GMM", "SGD (careful: tuned + averaging)"):
        r = per_model[nm]["0.500"]
        ch = {}
        for key, base in (("wd", good), ("wa", ~good), ("wd_ind", good), ("wa_ind", ~good)):
            counts = r["_" + key].sum(0).astype(float)
            order = np.argsort(-counts)
            k5 = max(1, int(round(0.05 * base.sum())))
            cand = np.flatnonzero(base)
            top_of_class = cand[np.argsort(-counts[cand])[:k5]]
            ch[key] = dict(events_per_model=float(counts.sum() / SEEDS), ever_affected=int((counts > 0).sum()),
                           share_in_top5pct_of_class=float(counts[top_of_class].sum() / max(counts.sum(), 1e-12)),
                           characteristics=characteristics(D, counts, base) if counts.sum() > 0 else [])
        out["methods"][nm]["characteristics"] = ch
    rng = np.random.default_rng(0)

    def boot(a, b):
        ds = []
        for _ in range(5000):
            j = rng.integers(0, SEEDS, SEEDS)
            ds.append(a[j].mean() - b[j].mean())
        return dict(diff=float(a.mean() - b.mean()), ci95=[float(np.quantile(ds, .025)), float(np.quantile(ds, .975))])

    out["paired"] = {}
    for other in ("SGD (textbook)", "SGD (same average as RA-GMM)", "SGD (careful: tuned + averaging)"):
        out["paired"][f"RA-GMM minus {other}"] = dict(
            wrongly_denied_n=boot(per_model["RA-GMM"]["0.500"]["_wd"].sum(1), per_model[other]["0.500"]["_wd"].sum(1)),
            wrongly_approved_n=boot(per_model["RA-GMM"]["0.500"]["_wa"].sum(1), per_model[other]["0.500"]["_wa"].sum(1)))
    (HERE / "error_types.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1)[:200])


if __name__ == "__main__":
    main()
