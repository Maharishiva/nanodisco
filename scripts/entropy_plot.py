"""Return and policy entropy on Breakout for rules trained frozen (rule_paper.py runs that logged entropy).

usage: python scripts/entropy_plot.py OUT.png RUN.npz:LABEL [RUN.npz:LABEL ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
COLORS = (INK, "#8a4fd8", "#2a78d6", "#1baf7a")
POINTS = 50
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
    }
)
fig, (ret_ax, ent_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
for (path, label), color in zip((a.rsplit(":", 1) for a in sys.argv[2:]), COLORS):
    z = np.load(path)
    log = z["log"][:, 0]  # first variant: [iters, seeds, 3]
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:])
    steps = (np.arange(n // w) + 1) * w * int(z["steps_per_iter"])
    ret = win[..., 0].sum(1) / np.maximum(win[..., 1].sum(1), 1)
    ent = win[..., 2].mean(1)
    for ax, y in ((ret_ax, ret), (ent_ax, ent)):
        ax.plot(steps, y, color=color, lw=0.8, alpha=0.35)
        ax.plot(steps, y.mean(1), color=color, lw=2.6, label=label)
ret_ax.set_title("Episode return (thin: one agent)", loc="left")
ent_ax.set_title("Policy entropy (nats; uniform over 3 actions = 1.10)", loc="left")
ent_ax.axhline(np.log(3), color=MUTED, lw=1, ls="--")
ent_ax.legend(loc="upper right", frameon=False)
for ax in (ret_ax, ent_ax):
    ax.set_xlabel("Environment steps per agent")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ent_ax.set_ylim(0, 1.2)
fig.suptitle("MinAtar Breakout, frozen rules, paper regime, lr 5e-4, 8 agents each", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
fig.savefig(sys.argv[1], dpi=200)
print("wrote", Path(sys.argv[1]))
