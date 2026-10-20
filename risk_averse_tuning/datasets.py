"""Bundled example data.

German Credit (Statlog), 1,000 applicants, 20 attributes, target ``creditability`` ("good" / "bad").
Source: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77 (CC BY 4.0). The copy is the decoded, human-readable version bundled in the PyPI
package scorecardpy 0.1.9.7 (see research/model_multiplicity/data/README.md for integrity checks). The dataset's cost
matrix: approving a bad applicant costs 5, rejecting a good one costs 1.
"""
from __future__ import annotations

import pathlib

import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent / "data"


def load_german_credit() -> pd.DataFrame:
    return pd.read_csv(DATA / "german_credit.csv")
