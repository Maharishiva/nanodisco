"""Every PPO and A2C learning rate against the frozen rules, lr 5e-4 (the rule's own Adam lr) highlighted.

usage: python scripts/bench_lr_plot.py RESULTS_DIR [ENV] [OUT.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
RULES = (
    ("rule_q", "Learned rule (value inputs)", "#2a78d6"),
    ("rule_noq", "Learned rule (no value inputs)", "#eb6834"),
)
BASES = (
    ("ppo", "PPO", "#1baf7a", "16 gradient steps per rollout"),
    ("a2c", "A2C", "#eda100", "1 gradient step per rollout"),
)
MATCH, POINTS = "0.0005", 60
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10,
    }
)

def curves(
    log, per_iter
):  # log [iters, variants, seeds, 2] -> steps, per-seed windowed returns [variants, seeds, POINTS]
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:]).sum(1)
    return (np.arange(n // w) + 1) * w * per_iter, (win[..., 0] / np.maximum(win[..., 1], 1)).transpose(1, 2, 0)

def sci(x):  # 0.00025 -> 2.5e-4
    mant, exp = f"{float(x):e}".split("e")
    return f"{float(mant):g}e{int(exp)}"

def tint(color, t):  # mix toward white: t=0 full color
    return tuple(c + (1 - c) * t for c in to_rgb(color))

def band(ax, steps, r, color, lw, label, z):
    mean, se = r.mean(0), r.std(0) / np.sqrt(r.shape[0])
    ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.15 if lw > 2 else 0, lw=0, zorder=z)
    return ax.plot(steps, mean, color=color, lw=lw, label=label, zorder=z + 1)[0]

root = Path(sys.argv[1])
env = sys.argv[2] if len(sys.argv) > 2 else "breakout"

def load(arm):
    return np.load(root / env / f"{arm}.npz")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
OTHERS = (
    ("rule_q", "Our rule, on-policy", "#2a78d6", ":", False),
    ("rule_q_paper", "Our rule, paper regime", "#2a78d6", "-", True),
    ("disco103", "Disco103, on-policy", "#e87ba4", ":", False),
    ("disco103_paper", "Disco103, paper regime", "#e87ba4", "-", True),
)
shared = {}
for ax, (arm, name, color, note) in zip(axes, BASES):
    z = load(arm)
    steps, r = curves(z["log"], int(z["steps_per_iter"]))
    lrs = [n.split("=")[1] for n in z["names"]]
    others = [i for i, lr in enumerate(lrs) if lr != MATCH]
    for t, i in zip((0.62, 0.45, 0.28), others):
        band(ax, steps, r[i], tint(color, t), 1.3, f"lr {sci(lrs[i])}", 2)
    band(ax, steps, r[lrs.index(MATCH)], color, 2.8, "lr 5e-4", 6)
    for key, label, c, ls, sweep in OTHERS:
        if not (root / env / f"{key}.npz").exists():
            continue
        zd = load(key)
        s, rr = curves(zd["log"], int(zd["steps_per_iter"]))
        i = int(np.argmax(rr[..., -max(1, rr.shape[-1] // 10) :].mean((-1, -2)))) if sweep else 0
        lr = sci(zd["names"][i].split("=")[1]) if sweep else ("5e-4" if key == "rule_q" else "3e-4")
        line = band(ax, s, rr[i], c, 2.8, "_", 8)
        line.set_linestyle(ls)
        shared[f"{label} (lr {lr})"] = line
    ax.set_title(f"{name}: {note}")
    ax.set_xlabel("Environment steps")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k")
    )
    ax.set_xlim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(title=name, loc="upper left", frameon=False, handlelength=2, fontsize=9.5, title_fontsize=10)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
leg = fig.legend(
    shared.values(), shared.keys(), loc="lower center", ncol=2, frameon=False, handlelength=3, fontsize=10.5
)
for line in leg.get_lines():
    line.set_linewidth(3)
axes[0].set_ylabel("Episode return")
fig.suptitle({"breakout": "MinAtar Breakout", "catch": "Catch", "reach": "Reach"}[env], fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2)
out = Path(sys.argv[3]) if len(sys.argv) > 3 else root / f"{env}_lrs.png"
fig.savefig(out, dpi=200)
print("wrote", out)
