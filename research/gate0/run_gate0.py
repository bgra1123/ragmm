"""
Gate 0: retrain-to-retrain decision instability across 6 public datasets. Design: PLAN.md (pre-registered).

Run from the repository root:
    python research/gate0/run_gate0.py                 # all datasets
    python research/gate0/run_gate0.py --only pima     # one dataset
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
import loan_experiment as le  # noqa: E402
from paper_tuning import ra_gmm_params  # noqa: E402

DATA = HERE / "data"
LAM, BATCH, K, SEEDS, THRESHOLD, ZETA = 0.01, 16, 2000, 200, 0.5, 0.05


# ---------------------------------------------------------------------- #
# Datasets                                                                 #
# ---------------------------------------------------------------------- #

def compas():
    df = pd.read_csv(DATA / "compas-scores-two-years.csv")
    df = df[(df.days_b_screening_arrest.between(-30, 30)) & (df.is_recid != -1) & (df.c_charge_degree != "O")
            & (df.score_text != "N/A")]
    y = df["two_year_recid"].to_numpy()
    X = df[["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "c_charge_degree"]]
    return X, y


def adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    df = pd.read_csv(DATA / "adult-all.csv", header=None, names=cols, skipinitialspace=True)
    y = df["income"].str.startswith(">50K").astype(int).to_numpy()
    return df.drop(columns=["income", "fnlwgt", "race", "sex"]), y


def bank():
    df = pd.read_csv(DATA / "bank-full.csv", sep=";")
    y = (df["y"] == "yes").astype(int).to_numpy()
    return df.drop(columns=["y", "duration"]), y


def pima():
    df = pd.read_csv(DATA / "diabetes.csv")
    return df.drop(columns=["Outcome"]), df["Outcome"].to_numpy()


def heart():
    df = pd.read_csv(DATA / "heart.csv", encoding="utf-8-sig")
    X = df.drop(columns=["target", "sex"])
    for c in ("cp", "restecg", "slope", "ca", "thal"):
        X[c] = X[c].astype(str)
    return X, df["target"].to_numpy()


DATASETS = {"german": None, "compas": compas, "adult": adult, "bank": bank, "pima": pima, "heart": heart}


def prepare(name, test_frac=0.3, max_test=2000, split_seed=0):
    if name == "german":
        return le.load_data()
    Xdf, y01 = DATASETS[name]()
    X = pd.get_dummies(Xdf, dtype=float).to_numpy()
    y = np.where(np.asarray(y01) == 1, 1.0, -1.0)
    n_test = min(max_test, int(round(test_frac * len(y))))
    rng = np.random.default_rng(split_seed)
    test_idx = []
    for label in (1.0, -1.0):
        idx = np.flatnonzero(y == label)
        test_idx.extend(rng.choice(idx, size=round(n_test * len(idx) / len(y)), replace=False))
    mask = np.zeros(len(y), bool)
    mask[test_idx] = True
    Xtr, Xte = X[~mask], X[mask]
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    return Xtr, y[~mask], Xte, y[mask]


# ---------------------------------------------------------------------- #
# Metrics                                                                  #
# ---------------------------------------------------------------------- #

def decision_metrics(W, Xte, yte, wstar):
    P = le.sigmoid(W @ Xte.T)
    flag = P >= THRESHOLD
    ideal = le.sigmoid(Xte @ wstar) >= THRESHOLD
    flips = (flag != ideal).mean(1)                        # share of rows per seed
    S = flag.shape[0]
    p = flag.mean(0)
    pairwise = float(np.mean(2 * p * (1 - p)) * S / (S - 1))
    aucs = np.array([le.auc(P[s], yte) for s in range(S)])
    return dict(flips_vs_exact_mean=float(flips.mean()), flips_vs_exact_p95=float(np.quantile(flips, 0.95)),
                retrain_disagreement=pairwise, not_unanimous=float(np.mean((p > 0) & (p < 1))),
                rows_flip_prob_ge_10pct=float(np.mean(np.minimum(p, 1 - p) >= 0.10)),
                auc_mean=float(aucs.mean()), auc_worst5=float(np.sort(aucs)[:max(1, S // 20)].mean()),
                flag_rate=float(flag.mean())), flips


def run(name):
    t0 = time.time()
    Xtr, ytr, Xte, yte = prepare(name)
    n, d = Xtr.shape
    L, mu = le.constants(Xtr, LAM)
    wstar, fstar = le.solve_exact(Xtr, ytr, LAM)
    kappa = L / mu
    m = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    alphas = [c / L for c in np.logspace(-2, np.log10(2.0), 12)]
    tuned_avg = le.tune_for_mean(Xtr, ytr, LAM, fstar, K, BATCH, L,
                                 [dict(alpha=a, beta=0.0, gamma=0.0, tail_average=True) for a in alphas])
    p = ra_gmm_params(L, mu, d, ZETA, 0.25)
    methods = {
        "SGD (textbook)": dict(alpha=1 / L, beta=0.0, gamma=0.0),
        "Nesterov (textbook)": dict(alpha=1 / L, beta=m, gamma=m),
        "SGD (lr=0.01)": dict(alpha=0.01, beta=0.0, gamma=0.0),
        "SGD + tail averaging (tuned)": tuned_avg,
        "RA-GMM (eps=0.25)": dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]),
    }
    idx = le.batch_indices(range(SEEDS), K, n, BATCH)
    ideal_flag = le.sigmoid(Xte @ wstar) >= THRESHOLD
    out = dict(dataset=name, n_train=n, n_test=int(len(yte)), d=d, L=L, mu=mu, kappa=kappa,
               positive_rate_test=float((yte > 0).mean()), exact_model_auc=float(le.auc(le.sigmoid(Xte @ wstar), yte)),
               exact_model_flag_rate=float(ideal_flag.mean()),
               epochs=K * BATCH / n, methods={}, params={k: {kk: float(vv) for kk, vv in v.items()} for k, v in methods.items()})
    flips = {}
    for nm, cfg in methods.items():
        W, _ = le.run_gmm(Xtr, ytr, LAM, idx, **cfg)
        if not np.all(np.isfinite(W)):
            out["methods"][nm] = dict(diverged=True)
            continue
        sub = le.objective(W, Xtr, ytr, LAM) - fstar
        met, flips[nm] = decision_metrics(W, Xte, yte, wstar)
        met["subopt_mean"], met["subopt_p95"] = float(sub.mean()), float(np.quantile(sub, 0.95))
        out["methods"][nm] = met
    ref = "SGD + tail averaging (tuned)"
    out["paired_vs_tuned_avg"] = {nm: le.paired_bootstrap(flips[nm], flips[ref], np.mean)
                                  for nm in flips if nm != ref}
    tb = [out["methods"][nm].get("flips_vs_exact_mean", 1.0) for nm in ("SGD (textbook)", "Nesterov (textbook)")]
    out["unstable_by_gate"] = bool(max(tb) >= 0.05)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        res = run(name)
        (outdir / f"{name}.json").write_text(json.dumps(res, indent=2))
        tb = res["methods"]
        print(f"{name:7s} n={res['n_train']:6d} d={res['d']:3d} kappa={res['kappa']:7.1f} epochs={res['epochs']:.2f} "
              f"AUC*={res['exact_model_auc']:.3f} | flips SGD {tb['SGD (textbook)'].get('flips_vs_exact_mean', float('nan')):.3f} "
              f"Nest {tb['Nesterov (textbook)'].get('flips_vs_exact_mean', float('nan')):.3f} "
              f"lr.01 {tb['SGD (lr=0.01)'].get('flips_vs_exact_mean', float('nan')):.3f} "
              f"tuned-avg {tb['SGD + tail averaging (tuned)']['flips_vs_exact_mean']:.3f} | gate {res['unstable_by_gate']} "
              f"[{res['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
