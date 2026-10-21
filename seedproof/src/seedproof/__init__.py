"""seedproof: measure and reduce how much a model's decisions depend on the training seed.

Builds on risk-averse optimization (Can & Gurbuzbalaban, 2025, Optimization Methods and Software 40(6); arXiv:2204.11292).
"""
__version__ = "0.1.0"

from .adapter import StabilityAdapter
from .check import StabilityResult, compare_stability, stability_check
from .datasets import load_german_credit
from .polish import iter_batches, polish_logistic, polish_logistic_streaming
from .tabular import TabularEncoder, binary_target, stratified_split

__all__ = ["StabilityAdapter", "StabilityResult", "stability_check", "compare_stability", "polish_logistic",
           "polish_logistic_streaming", "iter_batches", "TabularEncoder", "binary_target", "stratified_split",
           "load_german_credit"]
