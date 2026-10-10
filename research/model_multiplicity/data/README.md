# Data

## `german_credit.csv`: Statlog (German Credit Data)

- **What:** 1,000 real loan applications from a German bank. 20 applicant
  attributes plus the outcome (`creditability`: 700 `good`, 300 `bad` credit risks).
- **Original source:** Hofmann, H. (1994). *Statlog (German Credit Data)*.
  UCI Machine Learning Repository, dataset id 144.
  https://doi.org/10.24432/C5NC77 (license: CC BY 4.0).
- **Copy used here:** the human-readable version bundled in the PyPI package
  `scorecardpy==0.1.9.7` (`scorecardpy/data/germancredit.csv`), because the UCI
  host was not reachable from the environment where this was first run. The
  categorical codes (A11, A34, …) are already decoded into text.
- **Integrity checks:** 1,000 rows, 21 columns, 700/300 class split, no missing
  values. The first records match the UCI original (e.g. record 1: 6 months,
  amount 1169, age 67, good; record 2: 48 months, amount 5951, age 22, bad).
- **SHA-256:** `2c0bae00275c028fc853a1ea72cc7a68002c3f6876c41300c5c948711540c8c6`

The dataset documentation specifies a cost matrix: approving a bad applicant
costs 5, rejecting a good applicant costs 1. The experiment uses it to set the
approval threshold.
