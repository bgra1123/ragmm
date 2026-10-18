"""Analysis for the function-space polish. Plan: PLAN.md. Run: python research/boost/analyze.py"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
import loan_experiment as le  # noqa: E402

rng = np.random.default_rng(0)
NAMES = {"baseline": "XGBoost, 300 noisy rounds (baseline)", "polish_matched": "200 noisy + 100 deterministic rounds (budget matched)",
         "polish_extra": "300 noisy + 100 deterministic rounds", "polish_linesearch": "200 noisy + 100 deterministic, line-search steps",
         "polish_risk5": "200 noisy + 100 deterministic, risk-averse steps (θ = 5)"}


def sig(z):
    return 1 / (1 + np.exp(-z))


def avg_random(P, m=10, n_units=200):
    return P[np.stack([rng.choice(len(P), m, replace=False) for _ in range(n_units)])].mean(1)


def metrics(P, y):
    S = len(P)
    good = y > 0
    D = P >= 0.5
    p = D.mean(0)
    cons = P.mean(0) >= 0.5
    chg = (D != cons[None]).sum(1)
    k5 = max(1, int(round(0.05 * S)))
    aucs = np.array([le.auc(P[s], y) for s in range(0, S, max(1, S // 40))])
    ll = np.mean([np.logaddexp(0, -y * np.log(np.clip(P[s], 1e-9, 1 - 1e-9) / np.clip(1 - P[s], 1e-9, 1))).mean() for s in range(0, S, max(1, S // 40))])
    A = P >= 5 / 6
    wd, wa = (~A) & good[None], A & (~good)[None]
    cost = (5.0 * wa.sum(1) + wd.sum(1)) / len(y)
    return dict(units=int(S), disagreement=float(np.mean(2 * p * (1 - p)) * S / (S - 1)), coin_flip_share=float(np.mean(np.minimum(p, 1 - p) >= 0.1)),
                auc=float(aucs.mean()), logloss=float(ll), changed_mean=float(chg.mean()), changed_worst5=float(np.sort(chg)[-k5:].mean()),
                cost_833=float(cost.mean()), wrongly_denied_833=float(wd.sum(1).mean() / good.sum()), wrongly_approved_833=float(wa.sum(1).mean() / (~good).sum()),
                default_among_approved_833=float((wa.sum(1) / np.maximum(A.sum(1), 1)).mean()))


def main():
    data = [np.load(HERE / "results" / f"margins_split{s}.npz") for s in range(3)]
    y = np.concatenate([d["y_test"] for d in data])
    out = {}
    for k in NAMES:
        out[NAMES[k]] = metrics(sig(np.concatenate([d[k] for d in data], axis=1)), y)
    base = [sig(d["baseline"]) for d in data]
    out["averaging 10 seeds of the baseline (reference)"] = metrics(np.concatenate([avg_random(P) for P in base], axis=1), y)
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=2))
    for k, r in out.items():
        print(f"{k[:58]:58s} dis {r['disagreement']*100:5.1f}% coin {r['coin_flip_share']*100:5.1f}% chg {r['changed_mean']:5.1f}/{r['changed_worst5']:5.1f} AUC {r['auc']:.3f} LL {r['logloss']:.3f} | .833: WD {r['wrongly_denied_833']*100:4.1f}% WA {r['wrongly_approved_833']*100:4.1f}% def {r['default_among_approved_833']*100:4.1f}% cost {r['cost_833']:.3f}")


if __name__ == "__main__":
    main()
