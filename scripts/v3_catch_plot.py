"""mini v3 on Catch: meta-training curve, and fresh agents trained by the frozen rule vs the random rule.

usage: python scripts/v3_catch_plot.py RUN_DIR [OUT.png]
RUN_DIR: scripts/catch_frozen.py output for one variant (meta_training.npy, fresh_frozen_*.npy, summary.json).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
BLUE = "#2a78d6"
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

def windows(x):  # mean over POINTS equal windows; returns window ends (in items) and means
    w = max(1, len(x) // POINTS)
    n = len(x) // w * w
    return (np.arange(n // w) + 1) * w, x[:n].reshape(-1, w).mean(1)

run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
per_update = summary["envs_per_agent"] * summary["rollout"]
fig, (meta_ax, fresh_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
x, y = windows(np.load(run / "meta_training.npy"))
meta_ax.plot(x, y, color=BLUE, lw=2.4)
meta_ax.set_title(f"Meta-training: {summary['agents']} agents, random lifetimes", loc="left")
meta_ax.set_xlabel("Meta-steps (10 agent updates each)")
meta_ax.set_ylabel("Episode return")
for name, label, color, ls in (("fresh_frozen_learned", "Learned rule (frozen)", BLUE, "-"),
                               ("fresh_frozen_random", "Random rule (control)", MUTED, "--")):
    x, y = windows(np.load(run / f"{name}.npy"))
    fresh_ax.plot(x * per_update, y, color=color, lw=2.4, ls=ls, label=label)
fresh_ax.set_title(f"Fresh agents trained by the rule ({summary['agents']} agents)", loc="left")
fresh_ax.set_xlabel("Environment steps per agent")
fresh_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e3:g}k"))
fresh_ax.legend(loc="lower right", frameon=False)
for ax in (meta_ax, fresh_ax):
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("mini v3 on Catch: replay 90%, 10 updates per meta-step, Retrace, V-trace", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run.parent / "v3_catch.png"
fig.savefig(out, dpi=200)
print("wrote", out)
