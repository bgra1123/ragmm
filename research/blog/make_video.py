"""Render a ~50 s explainer video (no voice) for the blog post. Needs matplotlib and ffmpeg.

  python research/blog/make_video.py      # writes research/blog/video/same_applicant.mp4

Numbers come from FACT_CHECK.md. The dot grid animation is an illustration of retraining, not replayed data.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "video"
FPS, W, H = 30, 12.8, 7.2
NAVY, INK, MUTED, GREEN, RED, AMBER, TEAL = "#10172a", "#e6e9f1", "#98a2b8", "#4fbf8f", "#e2606a", "#f0a24b", "#5fb8c9"
SCENES = [("title", 6), ("retrain", 14), ("ever", 9), ("groups", 14), ("close", 8)]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
rng = np.random.default_rng(7)
N = 300
gx, gy = np.meshgrid(np.arange(25), np.arange(12))
gx, gy = gx.ravel()[:N], gy.ravel()[:N]
# illustration: base decision (79% approve, as the exact model) and a per-applicant flip chance whose mean gives ~55 changes per retrain
base = rng.random(N) < 0.79
flip_p = rng.beta(0.6, 2.4, N)
flip_p *= 55 / flip_p.sum()
flip_p = np.clip(flip_p, 0, 0.6)


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def text(ax, x, y, s, size, color=INK, weight="normal", alpha=1.0, ha="center"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, alpha=alpha, ha=ha, va="center", transform=ax.transAxes)


def grid(ax, colors, sizes, alpha=1.0):
    ax.scatter(0.155 + gx * 0.028, 0.70 - gy * 0.042, s=sizes, c=colors, alpha=alpha, transform=ax.transAxes, linewidths=0)


def frame_title(ax, t, T):
    a = ease(t / 1.2)
    text(ax, 0.5, 0.58, "Same applicant. Same data.", 40, weight="bold", alpha=a)
    text(ax, 0.5, 0.46, "Different answer.", 40, color=AMBER, weight="bold", alpha=ease((t - 1.2) / 1.0))
    text(ax, 0.5, 0.30, "How much of a credit decision depends on luck in training?", 18, MUTED, alpha=ease((t - 2.4) / 1.0))


state = {"dec": base.copy(), "ever": np.zeros(N, bool), "k": 0}


def frame_retrain(ax, t, T):
    if int(t * 4) != state["k"]:  # four retrains per second
        state["k"] = int(t * 4)
        state["dec"] = np.where(rng.random(N) < flip_p, ~base, base)
        state["ever"] |= state["dec"] != base
    diff = state["dec"] != base
    colors = np.where(diff, AMBER, np.where(state["dec"], GREEN, RED))
    grid(ax, colors, np.where(diff, 70, 45), alpha=ease(t / 0.8))
    text(ax, 0.5, 0.88, "300 applicants. The model is retrained with a new random seed each time.", 19, alpha=ease(t / 0.8))
    text(ax, 0.155, 0.13, f"retrain #{state['k'] + 1}", 15, MUTED, ha="left")
    text(ax, 0.845, 0.13, f"{diff.sum():>3d} decisions differ from the exact model", 15, AMBER, ha="right")
    text(ax, 0.5, 0.05, "Typical retrain: 55 of 300 decisions differ (18%)", 20, weight="bold", alpha=ease((t - 5) / 1.0))
    text(ax, 0.5, 0.97, "illustration of retraining; figures from 1,000 real retrains", 11, MUTED)
    legend = [(GREEN, "approve"), (RED, "deny"), (AMBER, "changed by the seed")]
    for i, (c, s) in enumerate(legend):
        ax.scatter([0.30 + 0.17 * i], [0.80], s=60, c=c, transform=ax.transAxes)
        text(ax, 0.31 + 0.17 * i, 0.80, s, 13, MUTED, ha="left")


def frame_ever(ax, t, T):
    # 284 of 300 change at least once in 1,000 retrains: fill the grid up to 284 amber
    order = np.argsort(-flip_p)
    k = int(284 * ease(t / 5))
    ever = np.zeros(N, bool)
    ever[order[:k]] = True
    grid(ax, np.where(ever, AMBER, np.where(base, GREEN, RED)), 45)
    text(ax, 0.5, 0.88, "Across 1,000 retrains", 22, MUTED)
    text(ax, 0.5, 0.08, f"{k} of 300 applicants got a different answer at least once", 22, weight="bold")


GROUP_ROWS = [("All applicants", 24.4, 30.4, None), ("Housing 'for free'", 27.4, 37.8, 31.5)]


def frame_groups(ax, t, T):
    text(ax, 0.5, 0.88, "Chance of a wrong decision", 26, weight="bold")
    text(ax, 0.5, 0.81, "average over retrains, 300 held-out applicants", 14, MUTED)
    g = ease(t / 2.5)
    a_on = ease((t - 6) / 1.5)
    for r, (name, exact, sgd, adapt) in enumerate(GROUP_ROWS):
        y0 = 0.62 - r * 0.30
        text(ax, 0.32, y0 + 0.035, name, 18, weight="bold", ha="left")
        bars = [("exactly trained model", exact, TEAL, 1.0), ("textbook training", sgd, AMBER, 1.0)]
        if adapt is not None:
            bars.append(("with adapter", adapt, GREEN, a_on))
        for j, (lab, v, c, al) in enumerate(bars):
            y = y0 - 0.075 * (j + 1) + 0.06
            w = 0.55 * (v / 45) * (g if j < 2 else a_on)
            ax.add_patch(plt.Rectangle((0.32, y - 0.025), w, 0.05, color=c, alpha=al, transform=ax.transAxes))
            text(ax, 0.31, y, lab, 12, MUTED, ha="right", alpha=al)
            if w > 0.01:
                text(ax, 0.33 + w, y, f"{v:.1f}%", 14, c, weight="bold", ha="left", alpha=al)
    text(ax, 0.5, 0.05, "+10.4 points for this group from training randomness alone; the adapter removed about 60% of it",
         15, alpha=ease((t - 8) / 1.5))


def frame_close(ax, t, T):
    text(ax, 0.5, 0.56, "Ask your model twice.", 40, AMBER, weight="bold", alpha=ease(t / 1.0))
    text(ax, 0.5, 0.44, "Measure how often it changes its mind, and for whom.", 20, alpha=ease((t - 1) / 1.0))
    text(ax, 0.5, 0.12, "German Credit (public dataset), logistic regression. Statistical behaviour of a model,", 12, MUTED, alpha=ease((t - 2) / 1.0))
    text(ax, 0.5, 0.08, "not an accusation of bias and not credit advice.", 12, MUTED, alpha=ease((t - 2) / 1.0))


DRAW = {"title": frame_title, "retrain": frame_retrain, "ever": frame_ever, "groups": frame_groups, "close": frame_close}


def main():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    path = OUT / "same_applicant.mp4"
    with writer.saving(fig, str(path), dpi=100):
        for name, T in SCENES:
            for f in range(int(T * FPS)):
                t = f / FPS
                ax.clear()
                ax.set_facecolor(NAVY)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                fade = min(ease(t / 0.4), ease((T - t) / 0.4))
                DRAW[name](ax, t, T)
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY, alpha=1 - fade, transform=ax.transAxes, zorder=10))
                writer.grab_frame(facecolor=NAVY)
            print(name, "done", flush=True)
    print(path)


if __name__ == "__main__":
    main()
