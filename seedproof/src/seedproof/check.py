"""
Measure how much a model's decisions depend on the training seed, overall and by group.

``stability_check`` retrains any model many times and reports disagreement between retrains, unstable rows and (with true
labels) the chance of a wrong decision, split into the part due to training randomness. ``compare_stability`` puts several
results side by side (for example without and with the adapter).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from ._util import DISCLAIMER, proba1 as _proba1



@dataclass
class StabilityResult:
    name: str
    summary: dict
    groups: dict = field(default_factory=dict)
    probabilities: Optional[np.ndarray] = None

    def to_dict(self):
        return dict(name=self.name, summary=self.summary, groups=self.groups, disclaimer=DISCLAIMER)

    def to_markdown(self):
        s = self.summary
        lines = [f"## {self.name}", "",
                 f"- Retrains: {s['retrains']}; rows checked: {s['rows']}; seconds per retrain: {s['fit_seconds']:.2f}",
                 f"- Chance that two retrains disagree on a row: {s['disagreement']:.1%}",
                 f"- Unstable rows (decision differs from the majority in at least 10% of retrains): {s['unstable_share']:.1%}",
                 f"- Decisions that differ from the majority, per retrain: {s['changes_mean']:.1f} on average, "
                 f"{s['changes_worst5']:.1f} in the worst 5% of retrains"]
        if s.get("wrong") is not None:
            lines.append(f"- Chance of a wrong decision: {s['wrong']:.1%} for one retrain, {s['wrong_majority']:.1%} "
                         f"for the majority vote ({s['wrong_excess']:+.1%} due to training randomness)")
        if self.groups:
            lines += ["", "| Group | Rows | Wrong (one retrain) | Wrong (majority) | Due to randomness | Unstable rows |",
                      "|---|---|---|---|---|---|"]
            for g, v in self.groups.items():
                wr = "-" if v.get("wrong") is None else f"{v['wrong']:.1%}"
                wm = "-" if v.get("wrong_majority") is None else f"{v['wrong_majority']:.1%}"
                ex = "-" if v.get("wrong_excess") is None else f"{v['wrong_excess']:+.1%}"
                lines.append(f"| {g} | {v['rows']} | {wr} | {wm} | {ex} | {v['unstable_share']:.1%} |")
        return "\n".join(lines)

    def statements(self, top=3):
        """Plain sentences for the groups with the largest part of wrong decisions due to training randomness."""
        rows = [(g, v) for g, v in self.groups.items() if (v.get("wrong_excess") or 0) > 0]
        rows.sort(key=lambda gv: -gv[1]["wrong_excess"])
        return [f"For rows in '{g}', a single retrain of {self.name} makes a wrong decision {v['wrong']:.1%} of the time; "
                f"the majority vote of {self.summary['retrains']} retrains does so {v['wrong_majority']:.1%} of the time, "
                f"so {100 * v['wrong_excess']:.1f} percentage points come from training randomness."
                for g, v in rows[:top]]


def _stats(D, truth):
    S = D.shape[0]
    p = D.mean(0)
    maj = p >= 0.5
    changes = (D != maj[None]).sum(1)
    q = np.quantile(changes, 0.95)
    out = dict(rows=int(D.shape[1]), disagreement=float(np.mean(2 * p * (1 - p)) * S / max(S - 1, 1)),
               unstable_share=float(np.mean(np.minimum(p, 1 - p) >= 0.10)),
               changes_mean=float(changes.mean()), changes_worst5=float(changes[changes >= q].mean()))
    if truth is not None:
        wrong = float((D != truth[None]).mean())
        wm = float((maj != truth).mean())
        out.update(wrong=wrong, wrong_majority=wm, wrong_excess=wrong - wm)
    return out


def stability_check(make_model: Callable[[int], object], X_train, y_train, X_test, y_test=None, seeds=20,
                    groups: Optional[dict] = None, threshold=0.5, name="model", min_group_size=15,
                    keep_probabilities=False) -> StabilityResult:
    """
    Retrain ``make_model(seed)`` for each seed on the same training data and measure its decisions on X_test.

    make_model : seed -> unfitted classifier (fit / predict_proba). Wrap the adapter here to measure it, e.g.
        ``lambda s: StabilityAdapter(XGBClassifier(subsample=0.8), random_state=s)``.
    y_test : optional true labels (enables the chance of a wrong decision). The reference for "due to randomness" is
        the majority decision over all retrains.
    groups : optional {name: boolean mask over the test rows}; groups smaller than ``min_group_size`` are skipped.
    """
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds")
    P, secs = [], []
    for s in seeds:
        t0 = time.perf_counter()
        m = make_model(s).fit(X_train, y_train)
        secs.append(time.perf_counter() - t0)
        P.append(_proba1(m, X_test))
    P = np.array(P)
    D = P >= threshold
    truth = None
    if y_test is not None:
        yt = np.asarray(y_test)
        positive = np.unique(np.asarray(y_train))[-1]
        truth = yt == positive
    summary = _stats(D, truth)
    summary.update(retrains=len(seeds), fit_seconds=float(np.mean(secs)), threshold=threshold)
    res_groups = {}
    for g, mask in (groups or {}).items():
        mask = np.asarray(mask, bool)
        if mask.sum() >= min_group_size:
            res_groups[g] = _stats(D[:, mask], None if truth is None else truth[mask])
    return StabilityResult(name=name, summary=summary, groups=res_groups,
                           probabilities=P if keep_probabilities else None)


def compare_stability(results: Iterable[StabilityResult], group=None) -> str:
    """Markdown table with one row per result (overall, or for one group)."""
    lines = ["| Model | Two retrains disagree | Unstable rows | Changes per retrain (avg / worst 5%) | "
             "Wrong decision (one retrain) | Due to randomness | Seconds per fit |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.summary if group is None else r.groups.get(group)
        if s is None:
            continue
        wr = "-" if s.get("wrong") is None else f"{s['wrong']:.1%}"
        ex = "-" if s.get("wrong_excess") is None else f"{s['wrong_excess']:+.1%}"
        lines.append(f"| {r.name} | {s['disagreement']:.1%} | {s['unstable_share']:.1%} | "
                     f"{s['changes_mean']:.1f} / {s['changes_worst5']:.1f} | {wr} | {ex} | "
                     f"{r.summary['fit_seconds']:.2f} |")
    return "\n".join(lines)
