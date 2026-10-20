"""Risk Averse Tuning - risk-averse training and tail diagnostics (RA-GMM, Can & Gürbüzbalaban 2025)."""

__version__ = "0.4.0"

from . import quadratic
from .adapter import StabilityAdapter, StabilityResult, compare_stability, polish_logistic, stability_check
from .datasets import load_german_credit
from .classifier import RiskAverseClassifier
from .risk import RAGMMBounds, RAGMMSolution
from .tabular import TabularEncoder
from .tail_report import TailReport, tail_report

__all__ = ["RAGMM", "GMM", "RAGMMBounds", "RAGMMSolution", "quadratic", "RiskAverseClassifier", "TabularEncoder",
           "tail_report", "TailReport", "StabilityAdapter", "StabilityResult", "stability_check",
           "compare_stability", "polish_logistic", "load_german_credit"]


def __getattr__(name):
    # the PyTorch optimizers are optional: import torch only when they are used
    if name in ("RAGMM", "GMM"):
        from . import optimizer
        return getattr(optimizer, name)
    raise AttributeError(f"module 'risk_averse_tuning' has no attribute {name!r}")
