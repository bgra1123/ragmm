"""
Figures for the rebalancing backtest. Run after backtest.py:
    python research/portfolio/make_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

PANELS = ["Nesterov (textbook)", "RA-GMM (eps=0.25)", "SGD + tail averaging (tuned)"]


def wealth_fans(arr, summary, out, long_only=False):
    dates = pd.to_datetime(arr["rebalance_dates"])
    ideal = np.exp(arr["ideal|log_wealth_monthly"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name, color in zip(axes, PANELS, SERIES):
        W = np.exp(arr[f"{name}|log_wealth_monthly"])
        for s in range(0, W.shape[0], 4):                       # 50 of the 200 seeds as thin lines
            ax.plot(dates, W[s], color=color, lw=0.6, alpha=0.25)
        lo, hi = np.quantile(W, [0.05, 0.95], axis=0)
        ax.plot(dates, lo, color=color, lw=1.5, ls=(0, (3, 2)))
        ax.plot(dates, hi, color=color, lw=1.5, ls=(0, (3, 2)))
        ax.plot(dates, ideal, color=INK, lw=2)
        m = summary["methods"][name]
        if long_only:
            lo_tw, hi_tw = m["terminal_wealth"]["p05"], m["terminal_wealth"]["p95"]
            sub = f"final wealth {lo_tw:.0f}x to {hi_tw:.0f}x (5th to 95th pct)"
        else:
            sub = f"ruined in {m['ruined']['mean']:.0%} of seeds"
        ax.set_title(f"{name}\n{sub}", fontsize=11, color=INK)
        ax.set_yscale("log")
        if not long_only:
            ax.set_ylim(1e-3, 1e3)
    axes[0].set_ylabel("wealth (start = 1, log scale)")
    fig.suptitle("Same strategy, same data, 200 random seeds of the solver. Black: exact solver. "
                 "Thin lines: 50 seeds. Dashed: 5th and 95th percentiles.", color=INK2, fontsize=10, y=0.01)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def phantom_strip(arr, summary, out):
    names = list(summary["methods"])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.2))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        v = arr[f"{name}|phantom"].mean(1) * 100           # per-seed average, % of wealth per rebalance
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=SERIES[0], alpha=0.45, linewidths=0)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ticks = [10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(ticks, [f"{t}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("phantom trades: % of wealth traded per rebalance only because of the solver's seed\n"
                  "(each dot = one seed's average over 328 rebalances; black = mean)")
    ax.grid(axis="y", visible=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_only", action="store_true")
    args = ap.parse_args()
    tag = "main_longonly" if args.long_only else "main"
    suffix = "_longonly" if args.long_only else ""
    res = HERE / "results"
    summary = json.loads((res / f"backtest_{tag}.json").read_text())
    arr = dict(np.load(res / f"backtest_{tag}_arrays.npz"))
    figdir = res / "figures"
    figdir.mkdir(exist_ok=True)
    wealth_fans(arr, summary, figdir / f"wealth_fans{suffix}.png", long_only=args.long_only)
    phantom_strip(arr, summary, figdir / f"phantom_trades{suffix}.png")
    print("figures written to", figdir)


if __name__ == "__main__":
    main()


def worst_case_figures(long_only=True):
    """Experiment P3: wealth fans and tail of phantom trades, average-tuned vs worst-case-tuned."""
    tag = "worstcase_longonly" if long_only else "worstcase"
    res = HERE / "results"
    s = json.loads((res / f"{tag}.json").read_text())
    arr = dict(np.load(res / f"{tag}_arrays.npz"))
    figdir = res / "figures"
    figdir.mkdir(exist_ok=True)
    panels = ["Nesterov (textbook)", "Average-tuned", "Worst-case-tuned (EV@R)"]
    dates = pd.to_datetime(arr["rebalance_dates"])
    ideal = np.exp(arr["ideal|log_wealth_monthly"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name, color in zip(axes, panels, SERIES):
        W = np.exp(arr[f"{name}|log_wealth_monthly"])
        for k in range(0, W.shape[0], 4):
            ax.plot(dates, W[k], color=color, lw=0.6, alpha=0.25)
        lo, hi = np.quantile(W, [0.05, 0.95], axis=0)
        ax.plot(dates, lo, color=color, lw=1.5, ls=(0, (3, 2)))
        ax.plot(dates, hi, color=color, lw=1.5, ls=(0, (3, 2)))
        ax.plot(dates, ideal, color=INK, lw=2)
        tw = s["methods"][name]["terminal_wealth"]
        label = name.replace("Nesterov (textbook)", "Default settings (textbook Nesterov)")
        ax.set_title(f"{label}\nfinal wealth {tw['p05']:.0f}x to {tw['p95']:.0f}x (5th to 95th pct)",
                     fontsize=11, color=INK)
        ax.set_yscale("log")
    axes[0].set_ylabel("wealth (start = 1, log scale)")
    fig.suptitle("Same strategy, same data, 200 random seeds of the solver. Black: exact solver. "
                 "Thin lines: 50 seeds. Dashed: 5th and 95th percentiles.", color=INK2, fontsize=10, y=0.01)
    fig.savefig(figdir / f"{tag}_fans.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Tail plot: share of rebalances with phantom trades above x (out-of-sample dates)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, color in zip(["Average-tuned", "Worst-case-tuned (EV@R)"], SERIES[1:]):
        v = np.sort(arr[f"{name}|phantom"][:, 24:].ravel() * 100)
        surv = 1 - np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, surv, color=color, lw=2)
        p95 = np.quantile(v, 0.95)
        ax.annotate(f"{name}: worst 5% start at {p95:.0f}%", (p95, 0.05), xytext=(8, 10 if color == SERIES[1] else -16),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0.05, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("phantom trades in one rebalance (% of wealth)")
    ax.set_ylabel("share of rebalances above this level")
    fig.savefig(figdir / f"{tag}_tail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", figdir / f"{tag}_fans.png", "and", figdir / f"{tag}_tail.png")
