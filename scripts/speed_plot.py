"""Faster frozen-rule eval on Breakout: what costs time, and what fewer/larger updates or higher lrs do to the return.

usage: python scripts/speed_plot.py DIR OUT.png
DIR: rule_paper.py outputs of the speed sweep (base, e32b96, e32b384_lr1e3, e64b768, lr_sweep .npz).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 9})
d = Path(sys.argv[1])
POINTS = 40

def curve(path, i=0):
    z = np.load(path)
    log = z["log"][:, i]
    w = len(log) // POINTS
    win = log[: w * POINTS].reshape(POINTS, w, *log.shape[1:])
    ret = (win[..., 0].sum(1) / np.maximum(win[..., 1].sum(1), 1)).mean(1)
    updates = (np.arange(POINTS) + 1) * w
    return updates, updates * int(z["steps_per_iter"]), ret

LINES = (("lr_sweep", 0, "8 envs, batch 96, lr 5e-4 (base)", INK, "-"),
         ("lr_sweep", 1, "8 envs, batch 96, lr 1e-3", "#2a78d6", "-"),
         ("lr_sweep", 2, "8 envs, batch 96, lr 2e-3", "#1baf7a", "-"),
         ("lr_sweep", 3, "8 envs, batch 96, lr 4e-3", "#d64545", "-"),
         ("e32b96", 0, "32 envs, batch 96, lr 5e-4", "#8a4fd8", "--"),
         ("e32b384_lr1e3", 0, "32 envs, batch 384, lr 1e-3", "#e08a1e", "--"),
         ("e64b768", 0, "64 envs, batch 768, lr 5e-4", MUTED, "--"))
fig, (step_ax, upd_ax, ms_ax) = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=(1.1, 1.1, 0.8)))
for name, i, label, color, ls in LINES:
    updates, steps, ret = curve(d / f"{name}.npz", i)
    step_ax.plot(steps, ret, color=color, ls=ls, lw=2, label=label)
    upd_ax.plot(updates, ret, color=color, ls=ls, lw=2)
step_ax.set_title("Return vs environment steps", loc="left")
step_ax.set_xlabel("Environment steps per agent")
step_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M"))
step_ax.legend(loc="upper left", frameon=False)
upd_ax.set_title("Same runs vs agent updates: progress follows updates", loc="left")
upd_ax.set_xlabel("Agent updates")
for ax in (step_ax, upd_ax):
    ax.set_ylabel("Episode return (mean of 8 agents)")
TIMES = (("unroll 1", 16.0), ("unroll 1 + CUDA graphs", 17.0), ("unroll 4", 12.4), ("unroll 4 + CUDA graphs", 12.4),
         ("unroll 29 (full)", 8.5), ("unroll 29 + CUDA graphs", 8.5))
ms_ax.barh(range(len(TIMES)), [t for _, t in TIMES], color=["#2a78d6" if "29" in n else "#b9b7b0" for n, _ in TIMES])
for i, (n, t) in enumerate(TIMES):
    ms_ax.text(t + 0.2, i, f"{t:.1f} ms", va="center", color=INK)
ms_ax.set_yticks(range(len(TIMES)), [n for n, _ in TIMES])
ms_ax.invert_yaxis()
ms_ax.set_xlabel("ms per agent update (8 agents in parallel)")
ms_ax.set_title("Cost per update (compile: 98 s, cached: 10 s)", loc="left")
for ax in (step_ax, upd_ax, ms_ax):
    ax.grid(axis="x" if ax is ms_ax else "y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Frozen headline rule on MinAtar Breakout: speed of the eval", x=0.01, ha="left", fontsize=13,
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2)
fig.savefig(sys.argv[2], dpi=170)
print("wrote", sys.argv[2])
