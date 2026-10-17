"""Group-level effects of training noise and of the polish. Plan: PLAN.md (pre-registered). Run: python research/groups/group_experiment.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
for sub in ("polish", "linesearch", "optimizers", "model_multiplicity", "gate0"):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parents[1]))
import loan_experiment as le  # noqa: E402
import linesearch_experiment as ls  # noqa: E402
import optimizer_experiments as oe  # noqa: E402
import polish_experiments as pe  # noqa: E402

LAM, BATCH, K1, SEEDS, SPLITS, B = 0.01, 16, 2000, 100, 5, 5
GROUPS = {
    "G1 unemployed now": lambda d: d.present_employment_since == "unemployed",
    "G2 job unemployed/unskilled non-resident": lambda d: d.job == "unemployed/ unskilled - non-resident",
    "G3 housing for free": lambda d: d.housing == "for free",
    "G4 age < 25": lambda d: d.age_in_years < 25,
    "G5 not a foreign worker": lambda d: d.foreign_worker == "no",
    "G6 female": lambda d: d.personal_status_and_sex.str.startswith("female"),
    "G7 checking account < 0 DM": lambda d: d.status_of_existing_checking_account == "... < 0 DM",
    "G8 large and long loan": lambda d: (d.credit_amount > 4000) & (d.duration_in_month > 24),
    "G9 no property": lambda d: d.property == "unknown / no property",
    "G10 male divorced/separated": lambda d: d.personal_status_and_sex == "male : divorced/separated",
}
CONFIGS = ["SGD", "SGD + adapter", "SGD + line search", "Adam", "Adam + adapter", "Adam + line search"]


def test_mask(y, seed, n_test=300):
    rng = np.random.default_rng(seed)
    idx_all = []
    for label in (1.0, -1.0):
        idx = np.flatnonzero(y == label)
        idx_all.extend(rng.choice(idx, size=round(n_test * len(idx) / len(y)), replace=False))
    m = np.zeros(len(y), bool)
    m[idx_all] = True
    return m


def main():
    df = pd.read_csv(HERE.parent / "model_multiplicity" / "data" / "german_credit.csv")
    yall = np.where(df[le.TARGET] == "good", 1.0, -1.0)
    rng_sel = np.random.default_rng(0)
    store = {g: {c: [] for c in CONFIGS + ["exact"]} for g in GROUPS}       # per group: lists of (S, members) decision arrays
    store_good = {g: [] for g in GROUPS}
    store_idx = {g: [] for g in GROUPS}
    others = {c: [] for c in CONFIGS + ["exact"]}
    others_good = []
    split0 = {}
    for sp in range(SPLITS):
        Xtr, ytr, Xte, yte = le.load_data(split_seed=sp, n_test=300)
        m = test_mask(yall, sp)
        raw = df[m].reset_index(drop=True)
        assert np.array_equal(np.where(raw[le.TARGET] == "good", 1.0, -1.0), yte)
        n, d = Xtr.shape
        L, mu = le.constants(Xtr, LAM)
        wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
        pstar = le.sigmoid(Xte @ wstar)
        idx1 = le.batch_indices(range(SEEDS), K1, n, BATCH)
        stage1 = {"SGD": le.run_gmm(Xtr, ytr, LAM, idx1, alpha=1 / L, beta=0.0, gamma=0.0)[0],
                  "Adam": oe.run_opt("Adam", Xtr, ytr, idx1, 0.01)}
        K2 = round(B * n / BATCH)
        idx2 = le.batch_indices(range(100_000, 100_000 + SEEDS), K2, n, BATCH)
        W = {}
        for nm, W1 in stage1.items():
            W[nm] = W1
            sel, _ = pe.select_polish(Xtr, ytr, W1, K2, rng_sel)
            W[f"{nm} + adapter"] = pe.run_from(W1, Xtr, ytr, idx2, **sel)
            W[f"{nm} + line search"] = ls.gd(W1, Xtr, ytr, B, "LS", 0, L)[0]
        dec = {c: (le.sigmoid(W[c] @ Xte.T) >= 0.5) for c in CONFIGS}
        dec["exact"] = np.repeat((pstar >= 0.5)[None], SEEDS, 0)
        good = yte > 0
        member = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
        for g, mk in member.items():
            for c in dec:
                store[g][c].append(dec[c][:, mk])
            store_good[g].append(good[mk])
            store_idx[g].append(np.flatnonzero(mk) + 1000 * sp)
        nonmember = ~np.any(np.stack(list(member.values())), axis=0)
        for c in dec:
            others[c].append(dec[c][:, nonmember])
        others_good.append(good[nonmember])
        if sp == 0:
            split0 = dict(raw=raw, dec=dec, good=good, member=member, pstar=pstar)
        print("split", sp, "done", flush=True)

    def stats(decs, goods, exact):
        D = np.concatenate(decs, axis=1)                 # (S, instances)
        E = np.concatenate(exact, axis=1)
        gd = np.concatenate(goods)
        flips = (D != E)
        majority_diff = np.minimum(D.mean(0), 1 - D.mean(0)) >= 0.10
        return dict(instances=int(D.shape[1]), flip_rate=float(flips.mean()),
                    wrongly_denied_rate=float((~D[:, gd]).mean()) if gd.any() else None,
                    exact_wrongly_denied_rate=float((~E[:, gd]).mean()) if gd.any() else None,
                    n_good=int(gd.sum()), coin_flip_share=float(majority_diff.mean())), flips, gd

    out = dict(seeds=SEEDS, splits=SPLITS, groups={})
    rng = np.random.default_rng(0)
    for g in GROUPS:
        entry = {}
        for c in CONFIGS:
            s, flips, gd = stats(store[g][c], store_good[g], store[g]["exact"])
            s["excess_wrongly_denied"] = None if s["wrongly_denied_rate"] is None else s["wrongly_denied_rate"] - s["exact_wrongly_denied_rate"]
            bs = []
            D = np.concatenate(store[g][c], axis=1)
            E = np.concatenate(store[g]["exact"], axis=1)
            for _ in range(2000):
                j = rng.integers(0, SEEDS, SEEDS)
                bs.append((D[j] != E[j]).mean())
            s["flip_rate_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
            entry[c] = s
        out["groups"][g] = entry
    out["non_members"] = {}
    for c in CONFIGS:
        s, _, _ = stats(others[c], [np.asarray(x) for x in others_good], others["exact"])
        out["non_members"][c] = s

    # showcase selection (pre-registered rule)
    cands = {g: out["groups"][g]["SGD"] for g in GROUPS if out["groups"][g]["SGD"]["instances"] >= 150
             and out["groups"][g]["SGD"]["excess_wrongly_denied"] is not None}
    showcase = max(cands, key=lambda g: (round(cands[g]["excess_wrongly_denied"], 6), cands[g]["flip_rate"]))
    out["showcase_group"] = showcase
    raw, dec, good, member, pstar = (split0[k] for k in ("raw", "dec", "good", "member", "pstar"))
    mk = member[showcase] & good & (pstar >= 0.5)
    deny_prob = 1 - dec["SGD"].mean(0)
    order = [i for i in np.argsort(-deny_prob) if mk[i]][:3]
    cols = ["status_of_existing_checking_account", "duration_in_month", "credit_history", "purpose", "credit_amount",
            "savings_account_and_bonds", "present_employment_since", "job", "housing", "property"]
    out["examples"] = [dict(row={c: (raw.loc[i, c].item() if hasattr(raw.loc[i, c], "item") else raw.loc[i, c]) for c in cols},
                            repaid=True, exact_probability_repay=float(pstar[i]), exact_decision="approve",
                            approval_share={c: float(dec[c][:, i].mean()) for c in CONFIGS}) for i in order]
    (HERE / "results" / "groups.json").write_text(json.dumps(out, indent=2, default=str))
    print("showcase:", showcase)


if __name__ == "__main__":
    main()
