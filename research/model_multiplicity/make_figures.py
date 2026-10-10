"""
Figures for the loan experiment. Run after loan_experiment.py:
    python research/model_multiplicity/make_figures.py --lam 0.01 --K 2000
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e9eef5", "#2a78d6", "#0b2f5c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def applicant_grid(arrays, methods, out, cols=25):
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.2))
    for ax, name in zip(np.atleast_1d(axes), methods):
        p = arrays[f"{name}|p_i"]
        n = len(p)
        rows = int(np.ceil(n / cols))
        xs, ys = np.arange(n) % cols, rows - 1 - np.arange(n) // cols
        order = np.argsort(p)          # sort so the grid reads from stable to unstable
        sc = ax.scatter(xs, ys, c=p[order], cmap=SEQ, vmin=0, vmax=0.5, s=46,
                        edgecolors=SURFACE, linewidths=1.0)
        ax.set_title(f"{name}\n{int((p > 0).sum())} of {n} applicants ever flip", fontsize=11, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("share of retrains where the decision differs\nfrom the ideal model", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Each dot is a real held-out loan applicant (German Credit). 200 retrains per method.",
                 color=INK2, fontsize=10, y=0.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def flips_strip(arrays, methods, out):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(methods) + 1.2))
    rng = np.random.default_rng(0)
    for i, name in enumerate(methods):
        F = arrays[f"{name}|F"]
        y = i + rng.uniform(-0.18, 0.18, len(F))
        ax.scatter(F, y, s=10, color=SERIES[0], alpha=0.45, linewidths=0)
        p95 = np.quantile(F, 0.95)
        ax.plot([F.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.plot([p95] * 2, [i - 0.3, i + 0.3], color=INK2, lw=2, ls=(0, (2, 2)))
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.set_xlabel("applicants (out of 300) whose decision differs from the ideal model, per retrain")
    ax.grid(axis="y", visible=False)
    ax.text(1.0, 1.02, "solid: mean   dashed: worst 5% threshold (95th pct)", transform=ax.transAxes,
            ha="right", color=INK2, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def tail_curves(curves, methods, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for color, name in zip(SERIES, methods):
        c = curves[name]
        ks = sorted(int(k) for k in c)
        p95 = [c[str(k)]["p95"] for k in ks]
        ax.plot(ks, p95, color=color, lw=2)
        ax.annotate(name, (ks[-1], p95[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training iterations")
    ax.set_ylabel("95th percentile of f(w_k) − f*  (worst 5% of runs)")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="0.01")
    ap.add_argument("--K", default="2000")
    ap.add_argument("--thr", default="0.8333")
    args = ap.parse_args()
    res = HERE / "results"
    summary = json.loads((res / f"summary_lam{args.lam}.json").read_text())
    run = summary["runs"][args.K]
    arrays = dict(np.load(res / f"arrays_lam{args.lam}_K{args.K}_thr{args.thr}.npz"))
    figdir = res / "figures"
    figdir.mkdir(exist_ok=True)

    main_ra = next(m for m in run["methods"] if m.startswith("RA-GMM"))
    grid_methods = ["Nesterov (textbook)", "Nesterov (tuned for mean)", main_ra]
    applicant_grid(arrays, grid_methods, figdir / f"applicant_grid_lam{args.lam}_K{args.K}.png")
    flips_strip(arrays, list(run["methods"]), figdir / f"flips_per_retrain_lam{args.lam}_K{args.K}.png")
    curve_methods = ["Nesterov (textbook)", "Nesterov (tuned for mean)", "SGD + tail averaging (tuned)", main_ra]
    tail_curves(run["curves"], curve_methods, figdir / f"tail_curves_lam{args.lam}_K{args.K}.png")
    print("figures written to", figdir)


if __name__ == "__main__":
    main()


def lambda_vs_tuning_figure():
    """Experiment 1b: seed-dependence vs decision quality, objective change (B) vs optimizer tuning (A)."""
    s = json.loads((HERE / "results" / "lambda_vs_tuning.json").read_text())
    colors = {"A": SERIES[0], "B": SERIES[1], "A+B": SERIES[2], "ref": "#8a8984"}
    labels = {"A": "A: tune the optimizer (λ = 0.01)", "B": "B: raise λ, textbook optimizer",
              "A+B": "A + B: raise λ and tune", "ref": "textbook optimizer, λ = 0.01"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for arm in ("ref", "B", "A+B", "A"):
        rows = [r for r in s["rows"] if r["arm"] == arm]
        xs = [r["flips_vs_reference"]["mean"] for r in rows]
        ys = [r["cost_per_applicant"]["mean"] for r in rows]
        ax.scatter(xs, ys, s=70, color=colors[arm], edgecolors=SURFACE, linewidths=2, label=labels[arm], zorder=3)
        for r, x, y in zip(rows, xs, ys):
            tag = r["method"].replace(" (tuned for mean)", " tuned").replace(" (tuned)", "").replace(" (textbook)", "")
            txt = tag if arm in ("A", "ref") else f"λ={r['lam']:g}, {tag}"
            if r["lam"] == 1.0:              # three overlapping points: one shared label
                if arm != "A+B":
                    continue
                txt = "λ=1, all optimizers"
            dx, dy = {("A", "SGD tuned"): (6, 8), ("A", "SGD + tail averaging"): (-20, -18),
                      ("B", "λ=0.3, SGD"): (6, -12)}.get((arm, txt), (6, 4))
            ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2)
    ref = s["reference"]
    ax.axhline(ref["cost"], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.99, ref["cost"], " intended model's cost", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("applicants (of 300) whose decision differs from the intended model, per retrain (mean)")
    ax.set_ylabel("expected cost per applicant (lower is better)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    out = HERE / "results" / "figures" / "lambda_vs_tuning.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
