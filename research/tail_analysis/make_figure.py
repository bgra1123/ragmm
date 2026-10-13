"""Figure: RA-GMM's worst-5% excess loss relative to SGD, per dataset. Run after tail_rows.py."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
NAMES = {"german": "German Credit", "compas": "COMPAS", "adult": "Adult income", "bank": "Bank marketing",
         "pima": "Pima diabetes", "heart": "Heart disease"}
rows = [(NAMES[k], json.loads((HERE / "results" / f"{k}.json").read_text())["overlap"]) for k in NAMES]
vs_text = [o["ragmm_over_sgd_textbook_excess_all_rows"] for _, o in rows]
vs_match = [o["ragmm_over_sgd_matched_excess_all_rows"] for _, o in rows]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
y = range(len(rows))
ax.axvline(1.0, color="#52514e", lw=1, ls="--")
ax.text(1.03, -0.55, "same as SGD", color="#52514e", fontsize=9, va="bottom")
for i in y:
    ax.plot([vs_text[i], vs_match[i]], [i, i], color="#c3c2b7", lw=2, zorder=1)
ax.scatter(vs_text, y, s=70, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2, zorder=3,
           label="vs textbook SGD (step 1/L)")
ax.scatter(vs_match, y, s=70, color="#eb6834", edgecolor="#fcfcfb", linewidth=2, zorder=3,
           label="vs SGD with the same average loss")
ax.set_yticks(list(y))
ax.set_yticklabels([n for n, _ in rows], color="#0b0b0b")
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.15, 2.0)
ax.set_xticks([0.2, 0.5, 1, 2])
ax.set_xticklabels(["0.2×", "0.5×", "1×", "2×"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("RA-GMM's extra loss in its worst 5% of retrains, relative to SGD (left = better)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e8e7e2", lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2, frameon=False, fontsize=9.5)
fig.suptitle("RA-GMM beats textbook SGD in the worst case, but not SGD run at the same average loss",
             x=0.02, ha="left", fontsize=11.5, color="#0b0b0b")
fig.tight_layout()
fig.savefig(HERE / "figures" / "ragmm_vs_sgd_tail.png", dpi=160)
print("wrote figures/ragmm_vs_sgd_tail.png")
