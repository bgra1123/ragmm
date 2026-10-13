"""
Turn a table (pandas DataFrame) into a design matrix for the trainers in ``classifier.py``.

* numeric columns: missing values -> training median;
* other columns: one-hot (missing -> its own category ``"<missing>"``; categories unseen in training -> all zeros);
* every column (numeric and one-hot) is then standardized with the training mean / std (as in the research
  experiments, so the package reproduces their numbers), and an intercept column is appended.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

MISSING = "<missing>"


class TabularEncoder:
    def __init__(self, drop: Optional[Iterable[str]] = None):
        self.drop = list(drop or [])

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        if df.shape[1] == 0:
            raise ValueError("no feature columns left after dropping")
        self.columns_ = list(df.columns)
        self.numeric_ = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        self.categorical_ = [c for c in df.columns if c not in self.numeric_]
        self.medians_ = {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in self.numeric_}
        self.categories_ = {c: sorted(self._cat(df[c]).unique().tolist()) for c in self.categorical_}
        raw = self._raw(df)
        self.mean_ = raw.mean(0)
        std = raw.std(0)
        self.std_ = np.where(std > 0, std, 1.0)
        self.feature_names_ = (list(self.numeric_) + [f"{c}={v}" for c in self.categorical_ for v in self.categories_[c]]
                               + ["(intercept)"])
        return self

    @staticmethod
    def _cat(s: pd.Series) -> pd.Series:
        return s.astype(object).where(s.notna(), MISSING).astype(str)

    def _numeric(self, df):
        if not self.numeric_:
            return np.zeros((len(df), 0))
        return np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).to_numpy(float)
                                for c in self.numeric_])

    def _raw(self, df):
        parts = [self._numeric(df)]
        for c in self.categorical_:
            vals = self._cat(df[c]).to_numpy()
            parts.append(np.column_stack([(vals == v).astype(float) for v in self.categories_[c]]))
        return np.hstack(parts)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing from the data: {missing}")
        return np.hstack([(self._raw(df) - self.mean_) / self.std_, np.ones((len(df), 1))])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def binary_target(y: pd.Series, positive=None) -> tuple[np.ndarray, object]:
    """Map a two-valued column to {+1, -1}. ``positive`` is the value that counts as +1 (default: the larger / last)."""
    vals = pd.unique(y.dropna())
    if y.isna().any():
        raise ValueError("the target column has missing values")
    if len(vals) != 2:
        raise ValueError(f"the target must have exactly two values, found {len(vals)}: {list(vals)[:5]}")
    if positive is None:
        positive = sorted(vals.tolist(), key=str)[-1] if not all(isinstance(v, (int, float, np.number)) for v in vals) else max(vals)
    if positive not in set(vals.tolist()):
        # allow "1" for 1 etc.
        match = [v for v in vals if str(v) == str(positive)]
        if not match:
            raise ValueError(f"positive value {positive!r} is not one of {list(vals)}")
        positive = match[0]
    return np.where(y.to_numpy() == positive, 1.0, -1.0), positive


def stratified_split(y: np.ndarray, test_size: float, seed: int = 0) -> np.ndarray:
    """Boolean mask of test rows, stratified by label (same rule as the research experiments)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n_test = int(round(test_size * len(y)))
    test = []
    for label in (1.0, -1.0):
        idx = np.flatnonzero(y == label)
        test.extend(rng.choice(idx, size=round(n_test * len(idx) / len(y)), replace=False))
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return mask
