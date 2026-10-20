"""Product benchmark (pre-registered in PLAN.md). Run: python research/product/benchmark.py [split ...]"""
import json
import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore")
from risk_averse_tuning import StabilityAdapter, TabularEncoder, load_german_credit, stability_check  # noqa: E402
from risk_averse_tuning.cli_adapter import make_factory  # noqa: E402
from risk_averse_tuning.tabular import stratified_split  # noqa: E402

SEEDS = 30
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


def configs():
    from sklearn.linear_model import SGDClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    f = {m: make_factory(m) for m in ("sgd", "xgboost", "lightgbm", "random_forest", "mlp")}
    out = {"SGD logistic": f["sgd"],
           "SGD logistic + adapter (polish)": lambda s: StabilityAdapter(f["sgd"](s), random_state=s),
           "SGD logistic + adapter (entropic step)": lambda s: StabilityAdapter(f["sgd"](s), step="entropic", random_state=s),
           "SGD logistic, 25x longer": lambda s: SGDClassifier(loss="log_loss", alpha=0.01, max_iter=125, tol=None, random_state=s)}
    big = {"xgboost": lambda s: XGBClassifier(n_estimators=2000, max_depth=3, learning_rate=0.01, subsample=0.8,
                                              colsample_bytree=0.8, random_state=s, verbosity=0, n_jobs=1),
           "lightgbm": lambda s: LGBMClassifier(n_estimators=2000, num_leaves=15, learning_rate=0.01, subsample=0.8,
                                                subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1, n_jobs=1),
           "random_forest": lambda s: RandomForestClassifier(n_estimators=2000, min_samples_leaf=3, random_state=s, n_jobs=1)}
    names = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random forest", "mlp": "MLP"}
    for m, nm in names.items():
        out[nm] = f[m]
        out[f"{nm} + adapter (average 10)"] = (lambda g: lambda s: StabilityAdapter(g(s), n_seeds=10, random_state=s))(f[m])
        if m in big:
            out[f"{nm}, 10x trees"] = big[m]
    return out


def run_split(sp):
    df = load_german_credit()
    y = np.where(df.creditability == "good", 1.0, -1.0)
    te = stratified_split(y, 0.3, seed=sp)
    feats = df.drop(columns=["creditability"])
    enc = TabularEncoder().fit(feats[~te])
    Xtr, Xte = enc.transform(feats[~te])[:, :-1], enc.transform(feats[te])[:, :-1]
    y01 = (y > 0).astype(int)
    raw = feats[te].reset_index(drop=True)
    groups = {g: f(raw).to_numpy() for g, f in GROUPS.items()}
    res = {}
    for name, make in configs().items():
        r = stability_check(make, Xtr, y01[~te], Xte, y01[te], seeds=SEEDS, groups=groups, min_group_size=1, name=name)
        res[name] = r.to_dict()
        print(sp, name, f"{r.summary['disagreement']:.3f}", f"{r.summary['fit_seconds']:.2f}s", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"split{sp}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    for sp in (sys.argv[1:] or range(5)):
        run_split(int(sp))
