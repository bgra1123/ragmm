# References

One numbered list for every piece of content (blog posts, YouTube scripts, TikTok captions). Cite as **[n]**.
Repository evidence is cited by file path; every number used in content is traced in [`FACT_SHEET.md`](FACT_SHEET.md).

## The method this work builds on

1. Can, B., & Gürbüzbalaban, M. (2025). Entropic risk-averse generalized momentum methods. *Optimization Methods and
   Software*, 40(6), 1535–1583. https://doi.org/10.1080/10556788.2025.2549356 · preprint arXiv:2204.11292
   (copy in [`papers/can_et_al_2025.pdf`](../papers/can_et_al_2025.pdf)).
2. Ahmadi-Javid, A. (2012). Entropic value-at-risk: A new coherent risk measure. *Journal of Optimization Theory and
   Applications*, 155(3), 1105–1123.
3. Can, B. (2025). The entropic risk-averse line search: choose the step length by minimizing the entropic risk of the per-row loss
   change along the full gradient. Proposal built on [1, 2]; implemented and evaluated in this repository
   ([`research/linesearch/`](../research/linesearch/RESULTS.md), [`research/risk_effect/`](../research/risk_effect/RESULTS.md)). Unpublished.

## The problem: one dataset, many models

4. Marx, C. T., Calmon, F. du P., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML 2020*.
5. Black, E., Raghavan, M., & Barocas, S. (2022). Model multiplicity: Opportunities, concerns, and solutions. *FAccT 2022*.
6. D'Amour, A., et al. (2022). Underspecification presents challenges for credibility in modern machine learning.
   *Journal of Machine Learning Research*, 23(226), 1–61.
7. Summers, C., & Dinneen, M. J. (2021). Nondeterminism and instability in neural network optimization. *ICML 2021*.
8. Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys 2021*.

## The fixes used in the adapter

9. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
10. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using
    deep ensembles. *NeurIPS 2017*.
11. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
    In *Advances in Large Margin Classifiers*, MIT Press.

## Data and software

12. Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0).
13. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
14. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
15. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

## Follow-up (LLMs)

16. Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.

## Evidence in this repository

| Short name | File | What it shows |
|---|---|---|
| Credit report | [`research/report/REPORT.md`](../research/report/REPORT.md) | 1,000 retrains of SGD logistic regression on German Credit; who is affected |
| Group statements | [`research/groups/CONCRETE.md`](../research/groups/CONCRETE.md) | chance of a wrong decision by type of applicant, with and without the polish |
| SOTA models | [`research/sota2/RESULTS.md`](../research/sota2/RESULTS.md) | XGBoost, LightGBM, random forest, MLP: which adapter works |
| Line search | [`research/linesearch/RESULTS.md`](../research/linesearch/RESULTS.md) | the entropic (risk-averse) line search vs exact line search |
| Product benchmark | [`research/product/RESULTS.md`](../research/product/RESULTS.md) | the shipped package, before/after, and equal-compute alternatives |
