"""Breakout in two regimes: matched on-policy (bench.py) vs Disco103's paper regime (*_paper.py), best lr per method.

usage: python scripts/bench_regime_plot.py RESULTS_DIR [OUT.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
POINTS = 60
METHODS = (
    ("rule_q", "Our rule (value inputs)", "#2a78d6"),
    ("symbolic", "Symbolic formula of our rule", "#eb6834"),
    ("disco103", "Disco103 (DeepMind)", "#e87ba4"),
    ("ppo", "PPO", "#1baf7a"),
    ("a2c", "A2C", "#eda100"),
)
REGIMES = (
    ("", "On-policy: 128 envs x 16 steps,\none update per rollout, no replay"),
    ("_paper", "Paper regime: 8 envs x 29 steps per update,\nbatches of 96 x 29, 90% replay"),
)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 11,
    }
)

def curves(
    log, per_iter
):  # log [iters, variants, seeds, 2] -> steps, per-seed windowed returns [variants, seeds, POINTS]
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:]).sum(1)
    return (np.arange(n // w) + 1) * w * per_iter, (win[..., 0] / np.maximum(win[..., 1], 1)).transpose(1, 2, 0)

def sci(x):
    mant, exp = f"{float(x):e}".split("e")
    return f"{float(mant):g}e{int(exp)}"

root = Path(sys.argv[1]) / "breakout"
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
handles = {}
for ax, (suffix, title) in zip(axes, REGIMES):
    ends = []
    for key, label, color in METHODS:
        f = root / f"{key}{suffix}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        steps, r = curves(z["log"], int(z["steps_per_iter"]))
        names = list(z["names"])
        sweep = "=" in names[0]
        i = int(np.argmax(r[..., -max(1, r.shape[-1] // 10) :].mean((-1, -2)))) if sweep else 0  # best lr
        mean, se = r[i].mean(0), r[i].std(0) / np.sqrt(r.shape[1])
        (handles[label],) = ax.plot(steps, mean, color=color, lw=2.4, zorder=3)
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.18, lw=0, zorder=2)
        lr = sci(names[i].split("=")[1]) if sweep else {"rule_q": "5e-4", "disco103": "3e-4", "symbolic": "5e-4"}[key]
        ends.append((mean[-1], f"lr {lr}", steps[-1]))
    gap, last = 2.2, -1e9
    for y, text, x in sorted(ends):  # learning-rate labels at line ends, nudged apart
        y = max(y, last + gap)
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
        last = y
    ax.set_title(title, loc="left")
    ax.set_xlabel("Environment steps")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
    ax.set_xlim(0, 5.6e6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Episode return")
order = [m[1] for m in METHODS if m[1] in handles]
leg = fig.legend(
    [handles[k] for k in order],
    order,
    loc="upper center",
    ncol=3,
    frameon=False,
    handlelength=2.6,
    bbox_to_anchor=(0.5, 0.93),
)
for line in leg.get_lines():
    line.set_linewidth(4)
fig.suptitle(
    "MinAtar Breakout, 8 seeds, best learning rate per method", x=0.01, ha="left", fontsize=13, fontweight="bold"
)
fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.5)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parent / "breakout_regimes.png"
fig.savefig(out, dpi=200)
print("wrote", out)
