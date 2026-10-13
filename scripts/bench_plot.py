"""Plot bench.py results: return of finished training episodes vs env steps, one panel per env.

usage: python scripts/bench_plot.py RESULTS_DIR [OUT.png]
Curves: mean over seeds, band = 1 standard error. A2C and PPO show their best learning rate
(by return over the last 10% of steps); every learning rate is printed.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ENVS = (("catch", "Catch"), ("reach", "Reach"), ("breakout", "MinAtar Breakout"))
SERIES = (
    ("rule_q", "Learned rule (value inputs)", "#2a78d6", "-"),
    ("rule_noq", "Learned rule (no value inputs)", "#eb6834", "-"),
    ("ppo", "PPO", "#1baf7a", "-"),
    ("a2c", "A2C", "#eda100", "-"),
    ("disco103", "Disco103 (DeepMind)", "#e87ba4", "-"),
    ("rule_untrained", "Untrained rule", "#8a8984", ":"),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
POINTS = 60
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
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
    ret = win[..., 0] / np.maximum(win[..., 1], 1)
    return (np.arange(n // w) + 1) * w * per_iter, ret.transpose(1, 2, 0)

def tail(r):  # per-variant mean over the last 10% of points
    return r[..., -max(1, r.shape[-1] // 10) :].mean((-1, -2))

def runs(env):  # series key -> (steps, per-seed returns), best learning rate for the baselines
    out = {}
    for arm in ("rule_q", "rule_noq", "ppo", "a2c", "disco103"):
        f = root / env / f"{arm}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        steps, r = curves(z["log"], int(z["steps_per_iter"]))
        names, scores = list(z["names"]), tail(r)
        for name, s in zip(names, scores):
            print(f"{env:8s} {name:22s} last 10%: {s:+.3f}")
        out[arm] = steps, r[int(np.argmax(scores)) if arm in ("ppo", "a2c") else 0]
        if arm == "rule_q":
            out["rule_untrained"] = steps, r[names.index("rule_untrained")]
    return out

steps_fmt = FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k")
root = Path(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
handles = {}
for ax, (env, title) in zip(axes, ENVS):
    data = runs(env)
    for key, label, color, ls in SERIES:
        if key not in data:
            continue
        steps, r = data[key]
        mean, se = r.mean(0), r.std(0) / np.sqrt(r.shape[0])
        (handles[label],) = ax.plot(steps, mean, color=color, lw=2.2, ls=ls, zorder=3)
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.18, lw=0, zorder=2)
    if not data:
        ax.text(0.5, 0.5, "running", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.xaxis.set_major_formatter(steps_fmt)
    ax.set_xlim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Episode return")
order = [s[1] for s in SERIES if s[1] in handles]
leg = fig.legend(
    [handles[k] for k in order],
    order,
    loc="upper center",
    ncol=4,
    frameon=False,
    handlelength=2.6,
    columnspacing=1.8,
    bbox_to_anchor=(0.5, 1.0),
)
for line in leg.get_lines():
    line.set_linewidth(4)
fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.5)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "bench.png"
fig.savefig(out, dpi=200)
print("wrote", out)
