"""Issue 1 figure: rules meta-trained with validation data from before vs after adaptation, trained frozen.

usage: python scripts/outer_plot.py OUTER_DIR BENCH_DIR [OUT.png]
OUTER_DIR: scripts/outer_check.py output (pre/, post/ with seedN.npz); BENCH_DIR: holds breakout/outer_rules.npz
from scripts/rule_paper.py, plus rule_q_paper.npz and symbolic_paper.npz for reference lines.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
VARIANTS = (
    ("pre", "Before adaptation (current)", "#2a78d6"),
    ("post", "After adaptation (on-policy)", "#c03a8f"),
)
CATCH_STEPS = 32 * (2 * 16 + 32)  # env steps per agent per frozen step: 2 rollouts to learn from + 1 held out
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

def windows(x, n):  # mean over n equal windows of the leading axis
    w = len(x) // n
    return x[: n * w].reshape(n, w, *x.shape[1:]).mean(1), w

def breakout(log):  # [iters, variants, seeds, 2] -> per-variant mean over seeds of windowed returns, window size
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:]).sum(1)
    return (win[..., 0] / np.maximum(win[..., 1], 1)).mean(-1).T, w

def style(ax, title):
    ax.set_title(title, loc="left")
    ax.set_xlabel("Environment steps per agent")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

outer, bench = Path(sys.argv[1]), Path(sys.argv[2]) / "breakout"
fig, (catch_ax, break_ax) = plt.subplots(1, 2, figsize=(12, 5.2))
z = np.load(bench / "outer_rules.npz")
names = [str(n).split()[1].split("/")[0] for n in z["names"]]
runs, w = breakout(z["log"])
steps = (np.arange(runs.shape[1]) + 1) * w * int(z["steps_per_iter"])
for variant, label, color in VARIANTS:
    fresh = np.stack([np.load(f)["fresh"] for f in sorted((outer / variant).glob("seed*.npz"))], 1)
    curves, cw = windows(fresh, POINTS)
    x = (np.arange(len(curves)) + 1) * cw * CATCH_STEPS
    catch_ax.plot(x, curves, color=color, lw=0.9, alpha=0.45)
    catch_ax.plot(x, curves.mean(1), color=color, lw=2.6)
    mine = runs[[i for i, n in enumerate(names) if n == variant]]
    break_ax.plot(steps, mine.T, color=color, lw=0.9, alpha=0.45)
    break_ax.plot(steps, mine.mean(0), color=color, lw=2.6)
    break_ax.annotate(f"{mine[:, -5:].mean():.0f}", (steps[-1], mine.mean(0)[-1]), xytext=(5, 0),
                      textcoords="offset points", va="center", fontsize=9.5, color=MUTED)
refs = (("rule_q_paper", "Earlier headline rule", INK, "--"),
        ("symbolic_paper", "Symbolic formula of it", "#eb6834", ":"))
for arm, label, color, ls in refs:
    r = np.load(bench / f"{arm}.npz")
    i = [str(n) for n in r["names"]].index(f"{arm} lr=0.0005")
    curve = breakout(r["log"][:, i : i + 1])[0][0]
    break_ax.plot(steps[: len(curve)], curve, color=color, lw=2, ls=ls)
style(catch_ax, "Catch: fresh agents, 8 per rule")
catch_ax.set_ylabel("Episode return")
style(break_ax, "MinAtar Breakout: paper regime, 8 seeds per rule")
break_ax.set_xlim(0, 5.5e6)
handles = [Line2D([], [], color=c, lw=4) for _, _, c in VARIANTS]
handles += [Line2D([], [], color=c, lw=2.4, ls=ls) for _, _, c, ls in refs]
handles.append(Line2D([], [], color=MUTED, lw=0.9, alpha=0.6))
labels = [v[1] for v in VARIANTS] + [r[1] for r in refs] + ["One rule (bold: mean of 4)"]
order = [0, 1, 4, 2, 3]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3, frameon=False,
           handlelength=2.6, bbox_to_anchor=(0.5, 0.925))
fig.suptitle("Outer-loss validation data collected before vs after the inner updates, rules then trained frozen",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5)
out = Path(sys.argv[3]) if len(sys.argv) > 3 else bench.parent / "outer_timing.png"
fig.savefig(out, dpi=200)
print("wrote", out)
