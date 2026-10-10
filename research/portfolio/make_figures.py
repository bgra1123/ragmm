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


def wealth_fans(arr, summary, out):
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
        ruin = summary["methods"][name]["ruined"]["mean"]
        ax.set_title(f"{name}\nruined in {ruin:.0%} of seeds", fontsize=11, color=INK)
        ax.set_yscale("log")
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
    res = HERE / "results"
    summary = json.loads((res / "backtest_main.json").read_text())
    arr = dict(np.load(res / "backtest_main_arrays.npz"))
    figdir = res / "figures"
    figdir.mkdir(exist_ok=True)
    wealth_fans(arr, summary, figdir / "wealth_fans.png")
    phantom_strip(arr, summary, figdir / "phantom_trades.png")
    print("figures written to", figdir)


if __name__ == "__main__":
    main()
