"""
Tail report: retrain a model many times and describe what happens in the worst runs and to which rows.

    from risk_averse_tuning import tail_report
    rep = tail_report(df, target="creditability", positive="good")
    print(rep.to_markdown())

Everything is computed on a held-out split. Definitions:

* a **retrain** is one training run with a different random seed (same data, same settings);
* the **tail runs** are the 5% of retrains with the highest training objective (the "bad days");
* a row's **decision changes** in a retrain if it differs from the decision of the exactly-trained model;
* an **unstable row** gets a different decision from the majority in at least 10% of retrains;
* the **chance of a wrong decision** for a group of rows is the average, over retrains and over the group's rows, of "the decision disagrees with the row's true label"; the exactly trained model gives the floor for
  the same rows, and the difference is the part **due to the randomness of training**;
* **leverage** (computed from one exact fit, no retraining) measures how much a row's prediction depends on directions
  of the model that the data pins down poorly; high-leverage rows are the ones retraining moves the most.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .classifier import METHODS, batch_indices, choose_parameters, objective, sigmoid, solve_exact, train_gmm
from .tabular import TabularEncoder, binary_target, stratified_split

NAMES = {"ragmm": "RA-GMM", "sgd": "SGD (step 1/L)", "careful": "SGD careful (tuned + averaging)"}
DISCLAIMER = ("This report describes how much a model's outputs vary between training runs and which rows are most "
              "affected. It is a statistical diagnostic, not a finding of bias, discrimination or wrongdoing by "
              "anyone, and not legal, credit or compliance advice.")


def empirical_evar(x, zeta):
    x = np.asarray(x, float)
    scale = max(float(np.mean(np.abs(x))), 1e-300)
    best = math.inf
    for t in np.geomspace(0.01, 200.0, 300) / scale:
        z = t * x
        m = z.max()
        best = min(best, (m + math.log(np.mean(np.exp(z - m))) - math.log(zeta)) / t)
    return float(best)


def leverage(X_train, y_train, w_star, lam, X_eval):
    """h_i = p_i (1 - p_i) x_i^T H^{-1} x_i / n with H the Hessian of the objective at w*."""
    n, d = X_train.shape
    s = sigmoid(y_train * (X_train @ w_star))
    H = (X_train.T * (s * (1 - s))) @ X_train / n + lam * np.eye(d)
    p = sigmoid(X_eval @ w_star)
    return p * (1 - p) * np.einsum("ij,ij->i", X_eval @ np.linalg.inv(H), X_eval) / n


def _auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class TailReport:
    summary: dict
    methods: dict
    tail_rows: dict
    at_risk_rows: pd.DataFrame
    notes: list = field(default_factory=list)
    group_risk: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(summary=self.summary, methods=self.methods, tail_rows=self.tail_rows, group_risk=self.group_risk,
                    at_risk_rows=self.at_risk_rows.to_dict(orient="records"), notes=self.notes, disclaimer=DISCLAIMER)

    def statements(self, method=None, top=3, min_rows=30):
        """Plain sentences about the chance of a wrong decision for the groups most affected by training randomness."""
        method = method or self.tail_rows["method"]
        rows = [(g, r) for g, r in self.group_risk[method].items()
                if g != "All held-out rows" and r["rows"] >= min_rows and r["excess"] > 0]
        rows.sort(key=lambda t: -t[1]["excess"])
        out = []
        for g, r in rows[:top]:
            out.append(f"For rows with {g} ({r['rows']} held-out rows), a model trained with {NAMES[method]} gives a wrong decision "
                       f"{r['p_wrong']:.1%} of the time (95% interval across retrains {r['p_wrong_ci95'][0]:.1%} to "
                       f"{r['p_wrong_ci95'][1]:.1%}). A model trained without randomness gives {r['p_wrong_exact']:.1%}; the other "
                       f"{r['excess'] * 100:.1f} percentage points come from the randomness of training.")
        return out

    def to_markdown(self) -> str:
        s = self.summary
        out = [f"# Tail report: {s['target']} (positive = {s['positive']!r})", "",
               f"{s['n_train']} training rows, {s['n_test']} held-out rows, {s['n_features']} features after encoding. "
               f"Each method was retrained {s['retrains']} times on the same data with different random seeds. "
               f"Decision: positive if the predicted probability is at least {s['threshold']}.",
               f"Exactly-trained model: held-out AUC {s['exact_auc']:.3f}, positive rate {s['exact_positive_rate']:.1%}.", "",
               "## How much do outputs change between training runs?", "",
               "| Method | Decisions changed per retrain (avg) | ... in the worst 5% of retrains | Rows with an unstable decision (changes in 10%+ of retrains) | Accuracy (avg / worst 5%) | Objective: avg / tail (CVaR 5%) / EV@R |",
               "|---|---|---|---|---|---|"]
        for nm, m in self.methods.items():
            out.append(f"| {NAMES[nm]} | {m['changed_mean']:.1f} ({m['changed_mean'] / s['n_test']:.1%}) | "
                       f"{m['changed_worst5']:.1f} | {m['coin_flip_rows']} ({m['coin_flip_rows'] / s['n_test']:.1%}) | "
                       f"{m['accuracy_mean']:.3f} / {m['accuracy_worst5']:.3f} | "
                       f"{m['objective_mean']:.4f} / {m['objective_cvar5']:.4f} / {m['objective_evar']:.4f} |")
        if self.group_risk:
            names = list(self.methods)
            out += ["", "## Chance of a wrong decision, by type of row", "",
                    "Chance that a retrained model gives the wrong decision for a row of this type (in brackets: percentage points "
                    "above a model trained without randomness, i.e. caused by the randomness of training).", "",
                    "| Type of row | Held-out rows | " + " | ".join(NAMES[m] for m in names) + " | Model without randomness |",
                    "|---|---|" + "---|" * (len(names) + 1)]
            for g in self.group_risk[names[0]]:
                r0 = self.group_risk[names[0]][g]
                out.append(f"| {g} | {r0['rows']}{' (indicative only)' if r0['rows'] < 60 else ''} | " + " | ".join(
                    f"{self.group_risk[m][g]['p_wrong']:.1%} ({self.group_risk[m][g]['excess'] * 100:+.1f})" for m in names)
                    + f" | {r0['p_wrong_exact']:.1%} |")
            out += [""] + [f"* {t}" for t in self.statements()] + [""]
        out += ["", "## Which rows are affected?", ""]
        tr = self.tail_rows
        out.append(f"Rows with the largest extra loss in the worst 5% of {NAMES[tr['method']]} retrains, compared "
                   f"with the other rows (largest differences first):")
        out += ["", "| Feature | Tail rows | Other rows |", "|---|---|---|"]
        for f in tr["profile"]:
            out.append(f"| {f['feature']} | {f['tail']:.3g} | {f['rest']:.3g} |")
        out += ["", f"These rows change decision in {tr['flip_prob_tail']:.0%} of retrains on average, other rows in "
                f"{tr['flip_prob_rest']:.0%}. Their leverage is {tr['leverage_ratio']:.1f}x the others' (median).", "",
                "## Rows at risk, found without retraining", "",
                f"The {len(self.at_risk_rows)} held-out rows with the highest leverage (one exact fit, no retraining) "
                "account for this share of all decision changes (a random selection of the same size would account "
                f"for about {len(self.at_risk_rows) / s['n_test']:.0%}):", ""]
        for nm, m in self.methods.items():
            auc = m["leverage_auc_for_coin_flip"]
            out.append(f"* {NAMES[nm]}: {m['changes_in_top_leverage']:.0%}"
                       + (f" (leverage identifies unstable rows with ROC AUC {auc:.2f})" if auc is not None else ""))
        out += ["", "Their row indices (in the input data) are in `at_risk_rows`.", ""]
        if self.notes:
            out += ["## Notes", ""] + [f"* {n}" for n in self.notes] + [""]
        out += ["---", f"*{DISCLAIMER}*"]
        return "\n".join(out)


def tail_report(df: pd.DataFrame, target: str, positive=None, drop: Optional[Iterable[str]] = None,
                methods: Iterable[str] = ("ragmm", "sgd", "careful"), retrains: int = 200, threshold: float = 0.5,
                test_size: float = 0.3, lam: float = 0.01, steps: int = 2000, batch_size: int = 16,
                zeta: float = 0.05, top_fraction: float = 0.10, seed: int = 0, groups: Optional[dict] = None,
                group_columns: Optional[Iterable[str]] = None, min_group_size: int = 30) -> TailReport:
    """
    ``groups`` maps a name to a function ``f(held_out_dataframe) -> boolean mask`` (any rule, including columns that are not model
    inputs such as age or sex); ``group_columns`` adds one group per value of each listed column that has at least
    ``min_group_size`` held-out rows. The report then states the chance of a wrong decision for each group.
    """
    methods = list(methods)
    bad = [m for m in methods if m not in METHODS]
    if bad or not methods:
        raise ValueError(f"methods must be a non-empty subset of {METHODS}, got {methods}")
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in the data")
    if retrains < 20:
        raise ValueError("retrains must be >= 20 so that the worst 5% contains at least one run")
    y, pos = binary_target(df[target], positive)
    test = stratified_split(y, test_size, seed)
    feats = df.drop(columns=[target])
    enc = TabularEncoder(drop=drop).fit(feats[~test])
    Xtr, Xte = enc.transform(feats[~test]), enc.transform(feats[test])
    ytr, yte = y[~test], y[test]
    w_star, f_star = solve_exact(Xtr, ytr, lam)
    p_star = sigmoid(Xte @ w_star)
    exact_dec = p_star >= threshold
    lev = leverage(Xtr, ytr, w_star, lam, Xte)
    k_top = max(1, int(round(top_fraction * len(yte))))
    top_lev = np.zeros(len(yte), bool)
    top_lev[np.argsort(-lev)[:k_top]] = True
    idx = batch_indices(range(seed, seed + retrains), steps, len(ytr), batch_size)

    out, per, decs = {}, {}, {}
    for m in methods:
        params, info = choose_parameters(m, Xtr, ytr, lam, steps, batch_size, zeta)
        W = train_gmm(Xtr, ytr, lam, idx, params["alpha"], params["beta"], params["gamma"], params["tail_average"])
        ok = np.all(np.isfinite(W), axis=1)
        W = W[ok]
        S = objective(W, Xtr, ytr, lam) - f_star
        tail = S > np.quantile(S, 0.95)
        P = sigmoid(W @ Xte.T)
        dec = P >= threshold
        changed = (dec != exact_dec[None]).sum(1)
        share = dec.mean(0)
        coin = np.minimum(share, 1 - share) >= 0.10
        flipprob = (dec != exact_dec[None]).mean(0)
        acc = (dec == (yte > 0)[None]).mean(1)
        k5 = max(1, len(W) // 20)
        excess = (np.logaddexp(0.0, -yte * (W[tail] @ Xte.T)) - np.logaddexp(0.0, -yte * (Xte @ w_star))).mean(0)
        out[m] = dict(params={k: float(v) for k, v in params.items()}, info=info, diverged_runs=int((~ok).sum()),
                      objective_mean=float(S.mean()), objective_p95=float(np.quantile(S, 0.95)),
                      objective_cvar5=float(S[tail].mean()) if tail.any() else float(S.max()),
                      objective_evar=empirical_evar(S, zeta),
                      changed_mean=float(changed.mean()), changed_worst5=float(np.sort(changed)[-k5:].mean()),
                      coin_flip_rows=int(coin.sum()), ever_changed_rows=int(((share > 0) & (share < 1)).sum()),
                      accuracy_mean=float(acc.mean()), accuracy_worst5=float(np.sort(acc)[:k5].mean()),
                      changes_in_top_leverage=float(flipprob[top_lev].sum() / max(flipprob.sum(), 1e-12)),
                      leverage_auc_for_coin_flip=_auc(lev, coin))
        per[m] = dict(excess=excess, flipprob=flipprob)
        decs[m] = dec

    # profile of the tail rows for the method with the most decision changes (the most informative tail)
    ref = max(methods, key=lambda m: out[m]["changed_mean"])
    k5r = max(1, int(round(0.05 * len(yte))))
    top = np.zeros(len(yte), bool)
    top[np.argsort(-per[ref]["excess"])[:k5r]] = True
    raw = feats[test].reset_index(drop=True)
    D = pd.get_dummies(raw.drop(columns=[c for c in (drop or []) if c in raw.columns]), dtype=float)
    smd = (D[top].mean() - D[~top].mean()) / np.sqrt(0.5 * (D[top].var() + D[~top].var()) + 1e-12)
    order = smd.abs().sort_values(ascending=False).index[:8]
    tail_rows = dict(method=ref, n_rows=int(top.sum()),
                     profile=[dict(feature=str(c), tail=float(D.loc[top, c].mean()), rest=float(D.loc[~top, c].mean()))
                              for c in order],
                     flip_prob_tail=float(per[ref]["flipprob"][top].mean()),
                     flip_prob_rest=float(per[ref]["flipprob"][~top].mean()),
                     leverage_ratio=float(np.median(lev[top]) / max(np.median(lev[~top]), 1e-300)),
                     row_index=[int(i) for i in np.flatnonzero(test)[top]])
    # chance of a wrong decision by group
    spec = {"All held-out rows": np.ones(len(yte), bool)}
    for c in (group_columns or []):
        if c not in raw.columns:
            raise ValueError(f"group column {c!r} not in the data")
        for v, cnt in raw[c].value_counts().items():
            if cnt >= min_group_size:
                spec[f"{c} = {v}"] = (raw[c] == v).to_numpy()
    for name, f in (groups or {}).items():
        spec[name] = np.asarray(f(raw), bool)
    truth = yte > 0
    gen = np.random.default_rng(seed)
    group_risk = {}
    for m in methods:
        D = decs[m]
        wrong = np.where(truth[None], ~D, D)
        wrong_exact = np.where(truth, ~exact_dec, exact_dec)
        res = {}
        for g, mk in spec.items():
            if mk.sum() == 0:
                continue
            w = wrong[:, mk]
            boots = [w[j].mean() for j in (gen.integers(0, len(w), len(w)) for _ in range(500))]
            pos_m, neg_m = mk & truth, mk & ~truth
            res[g] = dict(rows=int(mk.sum()), p_wrong=float(w.mean()), p_wrong_exact=float(wrong_exact[mk].mean()),
                          excess=float(w.mean() - wrong_exact[mk].mean()),
                          p_wrong_ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                          p_negative_given_positive=float((~D[:, pos_m]).mean()) if pos_m.any() else None,
                          p_positive_given_negative=float(D[:, neg_m].mean()) if neg_m.any() else None,
                          p_differs_from_exact=float((D[:, mk] != exact_dec[mk][None]).mean()))
        group_risk[m] = res
    test_index = np.flatnonzero(test)
    at_risk = pd.DataFrame(dict(row_index=test_index[top_lev], leverage=lev[top_lev],
                                exact_probability=p_star[top_lev])).sort_values("leverage", ascending=False)
    notes = []
    best = min(methods, key=lambda m: out[m]["changed_mean"])
    notes.append(f"Fewest decision changes per retrain: {NAMES[best]} ({out[best]['changed_mean']:.1f}).")
    if "ragmm" in out and "careful" in out and out["careful"]["changed_mean"] < out["ragmm"]["changed_mean"]:
        notes.append("The carefully tuned SGD changed fewer decisions than RA-GMM here, as in the research results "
                     "(research/tail_analysis/RESULTS.md); RA-GMM needs no tuning runs, the careful baseline needs 240.")
    if any(out[m]["diverged_runs"] for m in methods):
        notes.append("Some runs diverged and were excluded: " + ", ".join(
            f"{NAMES[m]}: {out[m]['diverged_runs']}" for m in methods if out[m]["diverged_runs"]))
    summary = dict(target=target, positive=pos if not isinstance(pos, np.generic) else pos.item(),
                   n_train=int(len(ytr)), n_test=int(len(yte)), n_features=int(Xtr.shape[1]),
                   retrains=retrains, threshold=threshold, lam=lam, steps=steps, batch_size=batch_size,
                   exact_auc=_auc(p_star, yte > 0), exact_positive_rate=float(exact_dec.mean()),
                   exact_accuracy=float((exact_dec == (yte > 0)).mean()))
    return TailReport(summary, out, tail_rows, at_risk, notes, group_risk)
