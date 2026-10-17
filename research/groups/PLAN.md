# Which groups of applicants are wrongly treated because of training noise, and does the adapter help them? (written before running)

Question (user): create concrete cases showing that responses were wrong for a specific group X, and improved with the additional adapter.

Rules to avoid cherry-picking: the groups, metrics, trainers and the **rule for choosing the showcase group and the example applicants are fixed here, before any result**.
Not a claim of bias or wrongdoing: group labels are used only to **describe** where training noise lands; sex, age and foreign-worker status are never inputs of any model (removed before training, as in all earlier experiments).

## Data and models

German Credit. **Five different stratified train/test splits** (seeds 0–4; 700 train / 300 held-out each), so each group has enough held-out members (pooled over splits; the same applicant can be held out in several splits).
Per split, 100 retrains (seeds 0–99) of two baselines with no adapter: **textbook SGD** (α = 1/L, 2,000 steps, batch 16) and **Adam (lr 0.01)**; each then polished with **5 epochs** by
(a) the **adapter** (step/momentum from the exact EV@R model, 6 pilots), (b) **GD exact line search** (deterministic). The reference is the exactly trained model of each split.
Decision rule 0.5 (primary) and 5/6 (secondary).

## Groups (fixed list; evaluation labels only)

G1 unemployed now · G2 job "unemployed / unskilled non-resident" · G3 housing "for free" · G4 age < 25 · G5 foreign worker = no · G6 sex (from personal status) = female ·
G7 checking account < 0 DM · G8 large and long loan (> 4,000 DM and > 24 months) · G9 no property · G10 male, divorced/separated.

## Metrics per group (pooled over retrains and splits; rule 0.5)

* **Flip rate**: share of (retrain, member) decisions that differ from the exact model's decision for that member.
* **Wrongly denied rate** among members who actually repaid, and the exact model's own rate for the same members; **excess** = baseline − exact.
* **Coin-flip share**: members whose decision differs from the majority in ≥ 10% of retrains.
* Same three numbers for non-members. Bootstrap (2,000 resamples of retrains, applicants held fixed) for the flip rate and the excess; it measures training noise only, not applicant sampling.

## Showcase selection (fixed rule)

* **Showcase group** = among groups with ≥ 150 member-instances, the one with the largest **excess wrongly-denied rate** under textbook SGD (ties: larger flip rate).
* **Examples** = in split 0, the three applicants of that group who repaid and whom the exact model approves, with the highest baseline probability of being denied across the 100 textbook retrains.
For each example: approval share across retrains for textbook SGD, SGD + adapter, SGD + line search, Adam, Adam + adapter; the exact decision.

## Hypotheses

* **H1** Training-induced flips are more frequent in the "unusual profile" groups (G1, G2, G3, G7, G8, G9) than among non-members: ratio ≥ 1.3 for at least 4 of the 6 (textbook SGD).
* **H2** In the showcase group the adapter cuts the flip rate and the excess wrongly-denied rate by ≥ 40%.
* **H3** The adapter reduces the flip rate in at least 8 of the 10 groups; the groups where it does not are reported.
* **H4** The deterministic line search cuts flips at least as much as the adapter in every group.
* **H5 (check)** After polishing, the showcase group's wrongly-denied rate approaches the exact model's own rate; it is not driven below it (the exact model's errors for this group remain).
