"""
StabilityAdapter: wrap a classifier so that retraining it with a different seed changes as few decisions as possible.

* strategy "polish": logistic / linear heads (SGDClassifier(loss="log_loss"), LogisticRegression). The objective is strongly
  convex, so the seed only decides where the optimizer stopped; a few deterministic steps move every run to the same point.
* strategy "average": models whose structure depends on the seed (XGBoost / LightGBM with subsampling, random forests,
  neural networks): average the predicted probabilities of ``n_seeds`` copies trained with different seeds.
* ``calibrate=True`` adds out-of-fold Platt scaling (needed when decisions use a cost-based cutoff, not 0.5).
"""
from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np

from ._util import proba1 as _proba1, sigmoid
from .polish import STEPS, iter_batches, platt_apply, platt_fit, polish_logistic, polish_logistic_streaming

STRATEGIES = ("auto", "polish", "average")



def _set_seed(est, seed, seed_param=None):
    est = _clone(est)
    if seed_param:
        est.set_params(**{seed_param: seed})
    elif hasattr(est, "get_params"):
        params = est.get_params()
        for k in ("random_state", "seed"):
            if k in params:
                est.set_params(**{k: seed})
                break
    elif hasattr(est, "seed"):
        est.seed = seed
    return est


def _clone(est):
    try:
        from sklearn.base import clone
        return clone(est)
    except Exception:
        return copy.deepcopy(est)


def _is_logistic(est):
    name = type(est).__name__
    if name == "LogisticRegression":
        return getattr(est, "penalty", "l2") in ("l2", None, "none")
    if name == "SGDClassifier":
        return est.loss in ("log_loss", "log") and est.penalty in ("l2", None)
    return False


class StabilityAdapter:
    """
    Wrap a classifier so that retraining it with a different seed changes as few decisions as possible.

        from sklearn.linear_model import SGDClassifier
        from xgboost import XGBClassifier
        StabilityAdapter(SGDClassifier(loss="log_loss", alpha=0.01)).fit(X, y)       # strategy "polish"
        StabilityAdapter(XGBClassifier(subsample=0.8), n_seeds=10).fit(X, y)          # strategy "average"

    Parameters
    ----------
    estimator : unfitted binary classifier with fit / predict_proba (scikit-learn API).
    strategy : "auto" (polish for logistic models, average otherwise), "polish" or "average".
    n_seeds : copies averaged by the "average" strategy.
    polish_epochs, step, theta : the polish (see ``polish_logistic``). lam : L2 strength of the polished objective;
        by default taken from the estimator (SGDClassifier alpha, LogisticRegression 1/(C n)).
    calibrate : add out-of-fold Platt scaling (``cv`` folds; costs cv extra fits of the whole adapter).
    random_state : base seed; copy i uses seed ``random_state * 1000 + i``.
    batch_size : if set, the polish reads the data in batches of this many rows (``polish_logistic_streaming``): the same
        result with the step search done from batch sums. For ``step="newton"`` the Hessian (d x d) is accumulated over batches.
    """

    def __init__(self, estimator, strategy="auto", n_seeds=10, polish_epochs=20, step="line_search", theta=1.0,
                 lam=None, calibrate=False, cv=5, random_state=0, seed_param=None, batch_size=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}, got {step!r}")
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        self.estimator, self.strategy, self.n_seeds = estimator, strategy, n_seeds
        self.polish_epochs, self.step, self.theta, self.lam = polish_epochs, step, theta, lam
        self.calibrate, self.cv, self.random_state, self.seed_param = calibrate, cv, random_state, seed_param
        self.batch_size = batch_size

    # -- fitting ---------------------------------------------------------------------------------------------------
    def _resolve(self):
        if self.strategy != "auto":
            return self.strategy
        return "polish" if _is_logistic(self.estimator) else "average"

    def _fit_core(self, X, y, seed):
        """Fit the uncalibrated adapter; return a function X -> p(positive)."""
        strategy = self._resolve()
        if strategy == "average":
            models = [_set_seed(self.estimator, seed * 1000 + i, self.seed_param).fit(X, y) for i in range(self.n_seeds)]
            return (lambda Z: np.mean([_proba1(m, Z) for m in models], axis=0)), dict(models=models)
        est = _set_seed(self.estimator, seed * 1000, self.seed_param).fit(X, y)
        name = type(est).__name__
        Xn = np.asarray(X, float)
        if len(est.classes_) != 2:
            raise ValueError("the polish strategy supports binary classification only")
        w0 = np.concatenate([np.ravel(est.coef_), np.ravel(est.intercept_)])
        F = np.column_stack([Xn, np.ones(len(Xn))])
        mask = np.concatenate([np.ones(Xn.shape[1]), [0.0]])
        yy = np.where(np.asarray(y) == est.classes_[1], 1.0, -1.0)
        if self.lam is not None:
            lam = self.lam
        elif name == "SGDClassifier":
            lam = est.alpha
        elif name == "LogisticRegression":
            lam = 1.0 / (est.C * len(yy)) if getattr(est, "penalty", "l2") == "l2" else 0.0
        else:
            lam = 1e-2
        lam = self.lam if self.lam is not None else lam
        if self.batch_size:
            w, info = polish_logistic_streaming(w0, iter_batches(F, yy, self.batch_size), lam, epochs=self.polish_epochs,
                                                step=self.step, theta=self.theta, mask=mask)
        else:
            w, info = polish_logistic(w0, F, yy, lam, epochs=self.polish_epochs, step=self.step, theta=self.theta, mask=mask)
        info.update(lam=lam)
        f = lambda Z: sigmoid(np.asarray(Z, float) @ w[:-1] + w[-1])  # noqa: E731
        return f, dict(model=est, coef=w, polish=info)

    def fit(self, X, y):
        t0 = time.perf_counter()
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("StabilityAdapter supports binary classification only")
        self.strategy_ = self._resolve()
        self._f, self.fitted_ = self._fit_core(X, y, self.random_state)
        n_fits = self.n_seeds if self.strategy_ == "average" else 1
        self.calibration_ = None
        if self.calibrate:
            y01 = (y == self.classes_[1]).astype(float)
            oof = np.zeros(len(y))
            folds = np.random.default_rng(self.random_state).permutation(len(y)) % self.cv
            Xa = np.asarray(X) if not hasattr(X, "iloc") else X
            for k in range(self.cv):
                tr, te = folds != k, folds == k
                sub = Xa.iloc[tr] if hasattr(Xa, "iloc") else Xa[tr]
                f_k, _ = self._fit_core(sub, y[tr], self.random_state)
                oof[te] = f_k(Xa.iloc[te] if hasattr(Xa, "iloc") else Xa[te])
            self.calibration_ = platt_fit(oof, y01)
            n_fits *= self.cv + 1
        self.info_ = dict(strategy=self.strategy_, fits=n_fits, fit_seconds=time.perf_counter() - t0,
                          calibrated=bool(self.calibrate))
        if "polish" in self.fitted_:
            self.info_["polish"] = self.fitted_["polish"]
        return self

    # -- prediction ------------------------------------------------------------------------------------------------
    def predict_proba(self, X):
        p = self._f(X)
        if self.calibration_ is not None:
            p = platt_apply(p, self.calibration_)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X)[:, 1] >= threshold, self.classes_[1], self.classes_[0])
