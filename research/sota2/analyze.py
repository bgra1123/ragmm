"""Analysis of saved probabilities. Plan: PLAN.md. Run: python research/sota2/analyze.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
for sub in ("groups", "model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
import loan_experiment as le  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

SPLITS = (0, 1, 2)
FAMILIES = ["XGBoost", "LightGBM", "Random forest", "HistGradientBoosting", "MLP"]
rng = np.random.default_rng(0)


def avg_random(P, m=10, n_units=200):
    idx = np.stack([rng.choice(len(P), m, replace=False) for _ in range(n_units)])
    return P[idx].mean(1)


def main():
    df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
    yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
    data = [np.load(HERE / "results" / f"probs_split{s}.npz") for s in SPLITS]
    y = np.concatenate([d["y_test"] for d in data])
    good = y > 0
    raws = [df[test_mask(yall, s)].reset_index(drop=True) for s in SPLITS]
    raw = pd.concat(raws, ignore_index=True)
    member = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
    nonmember = ~np.any(np.stack(list(member.values())), axis=0)

    configs = {}                                                   # name -> (units, applicants) pooled
    def add(name, per_split):
        configs[name] = np.concatenate(per_split, axis=1) if per_split[0].shape[0] == per_split[-1].shape[0] else None

    def get(key):
        return [d[key] for d in data]

    add("Logistic regression, exact", get("logistic exact|base"))
    for nm, key in (("Logistic SGD (no adapter)", "logistic SGD|base"), ("Logistic SGD + polish adapter", "logistic SGD|polish adapter"),
                    ("Logistic SGD + line search", "logistic SGD|line search")):
        add(nm, get(key))
    for fam in FAMILIES:
        base = get(f"{fam}|base")
        add(f"{fam} (no adapter)", base)
        add(f"{fam} + averaging (10 seeds)", [avg_random(P) for P in base])
        add(f"{fam} + calibration", get(f"{fam}|platt"))
        add(f"{fam} + calibration + averaging", [avg_random(P) for P in get(f"{fam}|platt")])
        if f"{fam}|refit" in data[0].files:
            add(f"{fam} + head refit", get(f"{fam}|refit"))
            add(f"{fam} + head refit + averaging", [avg_random(P) for P in get(f"{fam}|refit")])

    out = {}
    for name, P in configs.items():
        r = {}
        S = len(P)
        for thr in (0.5, 5 / 6):
            A = P >= thr
            wd, wa = (~A) & good[None], A & (~good)[None]
            cost = (5.0 * wa.sum(1) + wd.sum(1)) / len(y)
            k5 = max(1, int(round(0.05 * S)))
            r[f"{thr:.3f}"] = dict(wrongly_denied=float(wd.sum(1).mean() / good.sum()), wrongly_approved=float(wa.sum(1).mean() / (~good).sum()),
                                   default_among_approved=float((wa.sum(1) / np.maximum(A.sum(1), 1)).mean()), cost=float(cost.mean()),
                                   cost_cvar5=float(np.sort(cost)[-k5:].mean()), cost_p05=float(np.quantile(cost, .05)), cost_p95=float(np.quantile(cost, .95)))
        D = P >= 0.5
        p = D.mean(0)
        cons = (P.mean(0) >= 0.5)
        chg = (D != cons[None]).sum(1)
        k5 = max(1, int(round(0.05 * S)))
        aucs = np.array([le.auc(P[s], y) for s in range(0, S, max(1, S // 40))])
        r.update(units=int(S), disagreement=float(np.mean(2 * p * (1 - p)) * S / (S - 1)) if S > 1 else None,
                 coin_flip_share=float(np.mean(np.minimum(p, 1 - p) >= 0.10)), changed_mean=float(chg.mean()),
                 changed_worst5=float(np.sort(chg)[-k5:].mean()), auc=float(aucs.mean()))
        grp = {}
        flips = (D != cons[None])
        for g, mk in member.items():
            gg = good & mk
            grp[g] = dict(flip_rate=float(flips[:, mk].mean()), wrongly_denied=float((~D[:, gg]).mean()) if gg.any() else None)
        gg = good & nonmember
        grp["others"] = dict(flip_rate=float(flips[:, nonmember].mean()), wrongly_denied=float((~D[:, gg]).mean()))
        r["groups"] = grp
        out[name] = r
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=2))
    print("ok", len(out))


if __name__ == "__main__":
    main()
