"""
Numbers for the credit report (REPORT.md). Plan: PLAN.md in this folder.
Run from the repository root:  python research/report/credit_story.py
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
COLS = ["status_of_existing_checking_account", "duration_in_month", "credit_history", "purpose", "credit_amount",
        "savings_account_and_bonds", "present_employment_since", "other_debtors_or_guarantors", "property", "job",
        "housing"]


def roc_auc(score, label):
    return float(le.auc(score, np.where(label, 1.0, -1.0)))


def main():
    Xtr, ytr, Xte, yte = g0.prepare("german")
    Xdf, yall, train_mask = tr.raw_frame("german")
    raw_te = Xdf[~train_mask].reset_index(drop=True)
    raw_te = raw_te.assign(_y=yall[~train_mask])
    assert np.array_equal(raw_te["_y"].to_numpy(), yte)
    n, d = Xtr.shape
    L, mu = le.constants(Xtr, LAM)
    wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
    s = le.sigmoid(ytr * (Xtr @ wstar))
    H = (Xtr.T * (s * (1 - s))) @ Xtr / n + LAM * np.eye(d)
    p_star_te = le.sigmoid(Xte @ wstar)
    lev_te = p_star_te * (1 - p_star_te) * np.einsum("ij,ij->i", Xte @ np.linalg.inv(H), Xte) / n
    loss_star_te = np.logaddexp(0.0, -yte * (Xte @ wstar))

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
    out = dict(n_train=n, n_test=int(len(yte)), L=L, kappa=L / mu, exact_auc=float(le.auc(p_star_te, yte)),
               exact_approval_rate=float((p_star_te >= 0.5).mean()),
               params={k: {kk: float(v) for kk, v in c.items()} for k, c in methods.items()}, methods={})
    arrays = {}
    for nm, cfg in methods.items():
        W, _ = le.run_gmm(Xtr, ytr, LAM, idx, **cfg)
        S = le.objective(W, Xtr, ytr, LAM) - fstar
        tail = S > np.quantile(S, 0.95)
        P = le.sigmoid(W @ Xte.T)
        res = {}
        for thr in (0.5, 5 / 6):
            appr = P >= thr
            exact = p_star_te >= thr
            changed = (appr != exact).sum(1)                       # per run
            share = appr.mean(0)
            coin = np.minimum(share, 1 - share) >= 0.10
            res[f"{thr:.3f}"] = dict(
                changed_mean=float(changed.mean()), changed_worst5=float(np.sort(changed)[-SEEDS // 20:].mean()),
                changed_in_tail_runs=float(changed[tail].mean()), changed_in_other_runs=float(changed[~tail].mean()),
                approval_rate_tail=float(appr[tail].mean()), approval_rate_other=float(appr[~tail].mean()),
                coin_flip_applicants=int(coin.sum()), ever_changed=int(((share > 0) & (share < 1)).sum()))
            if thr == 0.5:
                arrays[nm] = dict(changed=changed, share=share, coin=coin, flipprob=(appr != exact).mean(0))
        acc = ((P >= 0.5) == (yte > 0)[None]).mean(1)
        Xtail = (np.logaddexp(0.0, -yte * (W[tail] @ Xte.T)) - loss_star_te).mean(0)
        arrays[nm]["Xtail"] = Xtail
        out["methods"][nm] = dict(
            objective_mean=float(S.mean()), objective_cvar5=float(S[tail].mean()),
            accuracy_tail_runs=float(acc[tail].mean()), accuracy_other_runs=float(acc[~tail].mean()),
            by_threshold=res)

    # tail applicants and leverage, per method
    rest_cols = {}
    for nm, a in arrays.items():
        top = np.zeros(len(yte), bool)
        top[np.argsort(-a["Xtail"])[:15]] = True
        flips_total = a["flipprob"].sum()
        top_lev = np.zeros(len(yte), bool)
        top_lev[np.argsort(-lev_te)[:30]] = True
        out["methods"][nm]["tail_applicants"] = dict(
            flip_prob_top=float(a["flipprob"][top].mean()), flip_prob_rest=float(a["flipprob"][~top].mean()),
            coin_flip_share_top=float(a["coin"][top].mean()), coin_flip_share_rest=float(a["coin"][~top].mean()),
            leverage_ratio=float(np.median(lev_te[top]) / np.median(lev_te[~top])))
        out["methods"][nm]["leverage_prediction"] = dict(
            share_of_changes_in_top10pct_leverage=float(a["flipprob"][top_lev].sum() / max(flips_total, 1e-12)),
            auc_leverage_for_coin_flip=roc_auc(lev_te, a["coin"]) if 0 < a["coin"].sum() < len(yte) else None,
            spearman_leverage_flipprob=float(pd.Series(lev_te).corr(pd.Series(a["flipprob"]), method="spearman")))
        if nm == "SGD (textbook)":
            D = pd.get_dummies(raw_te[COLS], dtype=float)
            smd = (D[top].mean() - D[~top].mean()) / np.sqrt(0.5 * (D[top].var() + D[~top].var()) + 1e-12)
            order = smd.abs().sort_values(ascending=False).index[:8]
            out["tail_applicant_profile"] = [dict(feature=c, top=float(D.loc[top, c].mean()),
                                                  rest=float(D.loc[~top, c].mean())) for c in order]
            out["tail_applicants_good_credit_share"] = [float((yte[top] > 0).mean()), float((yte[~top] > 0).mean())]
            out["tail_applicants_exact_prob_median"] = [float(np.median(p_star_te[top])), float(np.median(p_star_te[~top]))]

    # paired bootstrap of mean changed decisions
    rng = np.random.default_rng(0)
    def boot(a, b):
        diffs = [np.mean(a[j]) - np.mean(b[j]) for j in (rng.integers(0, SEEDS, SEEDS) for _ in range(5000))]
        return dict(diff=float(a.mean() - b.mean()), ci95=[float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))])
    ra_c = arrays["RA-GMM"]["changed"]
    out["paired_changed_decisions"] = {f"RA-GMM minus {k}": boot(ra_c, arrays[k]["changed"])
                                       for k in arrays if k != "RA-GMM"}

    # example applicants: most-flipped under textbook SGD
    ex = np.argsort(-arrays["SGD (textbook)"]["flipprob"])[:3]
    out["examples"] = [dict(**{c: (raw_te.loc[i, c].item() if hasattr(raw_te.loc[i, c], "item") else raw_te.loc[i, c]) for c in COLS},
                            actually_good_credit=bool(yte[i] > 0), exact_model_prob_good=float(p_star_te[i]),
                            exact_model_decision="approve" if p_star_te[i] >= 0.5 else "reject",
                            approval_share={k: float(a["share"][i]) for k, a in arrays.items()},
                            leverage_rank=int((lev_te > lev_te[i]).sum() + 1))
                       for i in ex]
    (HERE / "credit_story.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
