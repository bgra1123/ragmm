"""Function-space polish for XGBoost. Plan: PLAN.md. Run: python research/boost/boost_polish.py SPLIT"""
import pathlib
import sys
import time
import warnings

import numpy as np
import xgboost as xgb

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
for sub in ("model_multiplicity",):
    sys.path.insert(0, str(HERE.parent / sub))
sys.path.insert(0, str(HERE.parents[1]))
import loan_experiment as le  # noqa: E402

SEEDS = 100
NOISY = dict(max_depth=4, eta=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", nthread=1,
             objective="binary:logistic", verbosity=0)
DET = dict(max_depth=4, eta=0.1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, tree_method="hist", nthread=1,
           objective="binary:logistic", verbosity=0)
ALPHAS = np.geomspace(0.02, 2.0, 40)


def lme(z):
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).mean(-1, keepdims=True)))[..., 0]


def loss(F, t):                                   # logistic loss on margin F, t in {-1, +1}
    return np.logaddexp(0.0, -t * F)


def step(F, h, t, theta):
    base = loss(F, t)
    D = np.stack([loss(F + a * h, t) - base for a in ALPHAS])        # (A, n)
    obj = D.mean(1) if theta == 0 else lme(theta * D) / theta
    return ALPHAS[int(np.argmin(obj))]


def stage1(X, y01, Xte, seed, rounds):
    d = xgb.DMatrix(X, label=y01)
    b = xgb.train({**NOISY, "seed": seed}, d, rounds)
    return b.predict(d, output_margin=True), b.predict(xgb.DMatrix(Xte), output_margin=True)


def det_fixed(X, y01, Xte, F, Fte, rounds):
    d = xgb.DMatrix(X, label=y01, base_margin=F)
    b = xgb.train(DET, d, rounds)
    return b.predict(xgb.DMatrix(Xte, base_margin=Fte), output_margin=True)


def det_adaptive(X, y01, Xte, F, Fte, rounds, theta):
    t = 2.0 * y01 - 1.0
    F, Fte = F.copy(), Fte.copy()
    zeros, zte = np.zeros(len(y01)), np.zeros(len(Xte))
    for _ in range(rounds):
        b = xgb.train({**DET, "eta": 1.0}, xgb.DMatrix(X, label=y01, base_margin=F), 1)
        h = b.predict(xgb.DMatrix(X, base_margin=zeros), output_margin=True)
        hte = b.predict(xgb.DMatrix(Xte, base_margin=zte), output_margin=True)
        a = step(F, h, t, theta)
        F, Fte = F + a * h, Fte + a * hte
    return Fte


def main():
    split = int(sys.argv[1])
    t0 = time.time()
    Xtr, ytr, Xte, yte = le.load_data(split_seed=split, n_test=300)
    X, Xt = Xtr[:, :-1], Xte[:, :-1]
    y01 = (ytr > 0).astype(int)
    out = {k: [] for k in ("baseline", "polish_matched", "polish_extra", "polish_linesearch", "polish_risk5")}
    for s in range(SEEDS):
        F300, F300te = stage1(X, y01, Xt, s, 300)
        out["baseline"].append(F300te)
        F200, F200te = stage1(X, y01, Xt, s, 200)
        out["polish_matched"].append(det_fixed(X, y01, Xt, F200, F200te, 100))
        out["polish_extra"].append(det_fixed(X, y01, Xt, F300, F300te, 100))
        out["polish_linesearch"].append(det_adaptive(X, y01, Xt, F200, F200te, 100, 0))
        out["polish_risk5"].append(det_adaptive(X, y01, Xt, F200, F200te, 100, 5.0))
        if s % 20 == 0:
            print(split, s, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(HERE / "results" / f"margins_split{split}.npz", y_test=yte, **{k: np.stack(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
