"""Small shared helpers."""
import numpy as np

DISCLAIMER = ("These numbers describe how much a model's decisions vary between training runs, overall and by group. "
              "They are a statistical diagnostic, not a finding of bias or wrongdoing, and not legal, credit or "
              "compliance advice.")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def proba1(model, X):
    """Probability of the positive class from a scikit-learn style model."""
    return np.asarray(model.predict_proba(X))[:, 1]
