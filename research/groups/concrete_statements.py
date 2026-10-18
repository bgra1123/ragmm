"""Concrete chance-of-wrong-decision statements by category. Plan: PLAN.md addendum. Run: python research/groups/concrete_statements.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
for sub in ("polish", "linesearch", "model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parents[1]))
import loan_experiment as le  # noqa: E402
import linesearch_experiment as ls  # noqa: E402
import polish_experiments as pe  # noqa: E402
from group_experiment import GROUPS, test_mask  # noqa: E402

LAM, BATCH, K1, SEEDS, SPLITS, B = 0.01, 16, 2000, 100, 5, 5
THRS = (0.5, 5 / 6)
CONFIGS = ["SGD", "SGD + adapter", "SGD + line search"]


def main():
    df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
    yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
    rng_sel = np.random.default_rng(0)
    dec = {t: {c: [] for c in CONFIGS + ["exact"]} for t in THRS}
    goods, members = [], {g: [] for g in list(GROUPS) + ["All applicants"]}
    for sp in range(SPLITS):
        Xtr, ytr, Xte, yte = le.load_data(split_seed=sp, n_test=300)
        raw = df[test_mask(yall, sp)].reset_index(drop=True)
        n, d = Xtr.shape
        L, mu = le.constants(Xtr, LAM)
        wstar, _ = le.solve_exact(Xtr, ytr, LAM)
        idx1 = le.batch_indices(range(SEEDS), K1, n, BATCH)
        W1 = le.run_gmm(Xtr, ytr, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0]
        K2 = round(B * n / BATCH)
        sel, _ = pe.select_polish(Xtr, ytr, W1, K2, rng_sel)
        idx2 = le.batch_indices(range(100_000, 100_000 + SEEDS), K2, n, BATCH)
        W = {"SGD": W1, "SGD + adapter": pe.run_from(W1, Xtr, ytr, idx2, **sel), "SGD + line search": ls.gd(W1, Xtr, ytr, B, "LS", 0, L)[0]}
        for t in THRS:
            for c in CONFIGS:
                dec[t][c].append(le.sigmoid(W[c] @ Xte.T) >= t)
            dec[t]["exact"].append(np.repeat((le.sigmoid(Xte @ wstar) >= t)[None], SEEDS, 0))
        goods.append(yte > 0)
        for g, f in GROUPS.items():
            members[g].append(f(raw).to_numpy())
        members["All applicants"].append(np.ones(len(yte), bool))
        print("split", sp, flush=True)
    good = np.concatenate(goods)
    rng = np.random.default_rng(0)
    out = {"seeds": SEEDS, "splits": SPLITS, "groups": {}}
    for g, mk in members.items():
        mk = np.concatenate(mk)
        entry = dict(member_instances=int(mk.sum()), repaid=int((mk & good).sum()), defaulted=int((mk & ~good).sum()), rules={})
        for t in THRS:
            r = {}
            E = np.concatenate(dec[t]["exact"], axis=1)[:, mk]
            for c in CONFIGS + ["exact"]:
                D = np.concatenate(dec[t][c], axis=1)[:, mk]
                gm = good[mk]
                wrong = np.where(gm[None], ~D, D)                      # (S, members): denied-but-good or approved-but-bad
                r[c] = dict(p_wrong=float(wrong.mean()), p_wrongly_denied=float((~D[:, gm]).mean()) if gm.any() else None,
                            p_wrongly_approved=float(D[:, ~gm].mean()) if (~gm).any() else None, p_differs_from_exact=float((D != E).mean()))
                if c != "exact":
                    bs = [wrong[j].mean() for j in (rng.integers(0, SEEDS, SEEDS) for _ in range(1000))]
                    r[c]["p_wrong_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
            entry["rules"][f"{t:.3f}"] = r
        out["groups"][g] = entry
    (HERE / "results" / "concrete.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
