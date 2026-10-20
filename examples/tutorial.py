"""Code from docs/TUTORIAL.md, in order. Run: python examples/tutorial.py"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# --- Step 1: data -----------------------------------------------------------------------------------------------
from risk_averse_tuning import TabularEncoder, load_german_credit
from risk_averse_tuning.tabular import stratified_split

df = load_german_credit()
y = (df.pop("creditability") == "good").astype(int).to_numpy()   # 1 = repaid
test = stratified_split(2.0 * y - 1, 0.3, seed=0)                 # 300 held-out applicants
enc = TabularEncoder().fit(df[~test])
X_train = enc.transform(df[~test])[:, :-1]                         # drop the intercept column; sklearn fits its own
X_test = enc.transform(df[test])[:, :-1]
y_train, y_test = y[~test], y[test]

# --- Step 2: measure your current model ---------------------------------------------------------------------------
from xgboost import XGBClassifier
from risk_averse_tuning import stability_check


def make_xgb(seed):
    return XGBClassifier(n_estimators=200, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0, n_jobs=1)


groups = {"housing for free": (df[test].housing == "for free").to_numpy(),
          "no property": (df[test].property == "unknown / no property").to_numpy()}
before = stability_check(make_xgb, X_train, y_train, X_test, y_test, seeds=30, groups=groups, name="XGBoost")
print(before.to_markdown())

# --- Step 3: add the adapter -------------------------------------------------------------------------------------
from risk_averse_tuning import StabilityAdapter, compare_stability

model = StabilityAdapter(make_xgb(0), n_seeds=10).fit(X_train, y_train)
print(model.info_)                       # {'strategy': 'average', 'fits': 10, ...}
p_repay = model.predict_proba(X_test)[:, 1]

after = stability_check(lambda s: StabilityAdapter(make_xgb(s), n_seeds=10, random_state=s),
                        X_train, y_train, X_test, y_test, seeds=30, groups=groups, name="XGBoost + adapter")
print(compare_stability([before, after]))
print(compare_stability([before, after], group="housing for free"))

# --- Step 4: logistic models: polish instead of averaging -------------------------------------------------------
from sklearn.linear_model import SGDClassifier


def make_sgd(seed):
    return SGDClassifier(loss="log_loss", alpha=0.01, max_iter=5, tol=None, random_state=seed)


sgd = stability_check(make_sgd, X_train, y_train, X_test, y_test, seeds=30, name="SGD logistic")
pol = stability_check(lambda s: StabilityAdapter(make_sgd(s), random_state=s),
                      X_train, y_train, X_test, y_test, seeds=30, name="SGD + polish")
ent = stability_check(lambda s: StabilityAdapter(make_sgd(s), step="entropic", theta=1.0, random_state=s),
                      X_train, y_train, X_test, y_test, seeds=30, name="SGD + entropic polish")
print(compare_stability([sgd, pol, ent]))
print(StabilityAdapter(make_sgd(0)).fit(X_train, y_train).info_["polish"])

# --- Step 5: your own network's last layer ------------------------------------------------------------------------
from risk_averse_tuning import polish_logistic

features = np.column_stack([X_train, np.ones(len(X_train))])      # stand-in for penultimate activations + bias
w_head = np.zeros(features.shape[1])                              # stand-in for your trained last-layer weights
w_new, info = polish_logistic(w_head, features, 2.0 * y_train - 1, lam=0.01, epochs=20, step="line_search")
print(info)

# --- Step 6: cost-based cutoffs need calibration ------------------------------------------------------------------
cal = StabilityAdapter(make_xgb(0), n_seeds=10, calibrate=True).fit(X_train, y_train)
approve = cal.predict_proba(X_test)[:, 1] >= 5 / 6                 # German Credit cost matrix: 5 vs 1
print(f"approved {approve.mean():.1%} of held-out applicants at the 5/6 cutoff")

# --- Step 7: plain sentences for a report -------------------------------------------------------------------------
for s in before.statements():
    print("-", s)
