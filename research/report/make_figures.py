"""Figures for REPORT.md. Run after credit_story.py:  python research/report/make_figures.py"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
o = json.loads((HERE / "credit_story.json").read_text())
names = ["SGD (textbook)", "RA-GMM", "SGD (same average as RA-GMM)", "SGD (careful: tuned + averaging)"]
labels = ["Textbook SGD", "RA-GMM (paper)", "SGD, smaller step\n(same average as RA-GMM)", "Carefully tuned SGD"]
n = o["n_test"]
rows = []
for nm in names:
    b = o["methods"][nm]["by_threshold"]["0.500"]
    coin, ever = b["coin_flip_applicants"], b["ever_changed"]
    rows.append((n - ever, ever - coin, coin))

INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e2"
colors = ["#cde2fb", "#6da7ec", "#1c5cab"]
seg_names = ["Never changes", "Changes in under 10% of retrains", "Coin-flip: changes in 10% or more"]
fig, ax = plt.subplots(figsize=(9, 4.3), facecolor=SURF)
ax.set_facecolor(SURF)
for i, r in enumerate(rows):
    left = 0
    for j, v in enumerate(r):
        ax.barh(i, v, left=left, color=colors[j], edgecolor=SURF, linewidth=2, height=0.62,
                label=seg_names[j] if i == 0 else None)
        if v >= 14:
            ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=10,
                    color="#ffffff" if j == 2 else INK)
        left += v
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, n)
ax.set_xlabel(f"Held-out loan applicants (of {n}), each model retrained 1,000 times", color=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.16), ncol=3, frameon=False, fontsize=8.8)
fig.suptitle("How many applicants' loan decisions depend on the training run", x=0.02, ha="left",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(HERE / "figures" / "decisions_by_method.png", dpi=160)
print("wrote figures/decisions_by_method.png")
