"""Overview of many meta-trained rules on MinAtar Breakout: score per rule, then curves and entropy of the best ones.

usage: python scripts/night_plot.py OUT.png RUN_DIR [RUN_DIR ...] [--ref NPZ:LABEL ...] [--top 6] [--title T]
RUN_DIR holds breakout/breakout.npz from scripts/rule_paper.py (one variant per eval lr); the label is the dir name.
--ref: other rule_paper npz files shown as reference groups (e.g. the headline rule).
Prints one line per rule and lr: last-10% return mean, s.e., and entropy.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#ebeae6", "#b9b7b0"
COLORS = ("#2a78d6", "#8a4fd8", "#1baf7a", "#e08a1e", "#d64545", "#0b0b0b", "#7a7a7a", "#c45cb0")
POINTS = 50
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 9})
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("runs", type=Path, nargs="*")
p.add_argument("--ref", nargs="*", default=[])
p.add_argument("--top", type=int, default=6)
p.add_argument("--title", default="Meta-trained rules on MinAtar Breakout (frozen, paper regime, 8 agents per bar)")
a = p.parse_args()

def arms(path, label):  # one arm per eval lr: (label, per-agent last-10% returns, curves, entropy curves, steps)
    z = np.load(path)
    log, spi = z["log"], int(z["steps_per_iter"])
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:])
    steps = (np.arange(n // w) + 1) * w * spi
    tail = log[-max(1, len(log) // 10):].sum(0)
    out = []
    for i, name in enumerate(z["names"]):
        lr = str(name).split("lr=")[-1]
        ret = win[:, :, i, :, 0].sum(1) / np.maximum(win[:, :, i, :, 1].sum(1), 1)
        ent = win[:, :, i, :, 2].mean(1) if log.shape[-1] > 2 else None
        final = tail[i, :, 0] / np.maximum(tail[i, :, 1], 1)
        out.append(dict(label=f"{label} lr={lr}", rule=label, final=final, ret=ret, ent=ent, steps=steps))
    return out

found = [r for r in a.runs if (r / "breakout" / "breakout.npz").exists()]
rows = [x for r in found for x in arms(r / "breakout" / "breakout.npz", r.name)]
refs = [x for s in a.ref for x in arms(*s.rsplit(":", 1))]
best = {}
for x in rows:  # each rule's best eval lr
    if x["rule"] not in best or x["final"].mean() > best[x["rule"]]["final"].mean():
        best[x["rule"]] = x
order = sorted(best.values(), key=lambda x: -x["final"].mean())
for x in sorted(rows + refs, key=lambda x: -x["final"].mean()):
    ent = "" if x["ent"] is None else f"  entropy end {x['ent'][-1].mean():.2f}"
    print(f"{x['label']:40s} {x['final'].mean():6.1f} +- {x['final'].std() / np.sqrt(len(x['final'])):4.1f}{ent}")
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=(1.1, 1))
bar_ax, ret_ax, ent_ax = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
groups = refs + order
rng = np.random.default_rng(0)
for i, x in enumerate(groups):
    ref = x in refs
    color = REF if ref else COLORS[0]
    bar_ax.bar(i, x["final"].mean(), width=0.7, color=color, alpha=0.35)
    jitter = i + rng.uniform(-0.18, 0.18, len(x["final"]))
    bar_ax.scatter(jitter, x["final"], s=14, color=MUTED if ref else INK, zorder=3)
bar_ax.set_xticks(range(len(groups)), [x["label"] for x in groups], rotation=35, ha="right", fontsize=8.5)
bar_ax.set_ylabel("Return, last 10% of 5M steps")
bar_ax.set_title("Each rule at its best eval lr (grey: reference rules; dots: one agent each)", loc="left")
for x, color in zip(order[: a.top], COLORS):
    ret_ax.plot(x["steps"], x["ret"].mean(1), color=color, lw=2.2, label=x["label"])
    if x["ent"] is not None:
        ent_ax.plot(x["steps"], x["ent"].mean(1), color=color, lw=2.2, label=x["label"])
for x, ls in zip(refs, ("--", ":", "-.")):
    ret_ax.plot(x["steps"], x["ret"].mean(1), color=MUTED, lw=2, ls=ls, label=x["label"])
    if x["ent"] is not None:
        ent_ax.plot(x["steps"], x["ent"].mean(1), color=MUTED, lw=2, ls=ls)
ret_ax.set_title(f"Top {a.top}: episode return (mean of 8 agents)", loc="left")
ent_ax.set_title("Top rules: policy entropy (uniform over 3 actions = 1.10)", loc="left")
ret_ax.legend(loc="upper left", frameon=False)
ent_ax.set_ylim(0, 1.2)
for ax in (bar_ax, ret_ax, ent_ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in (ret_ax, ent_ax):
    ax.set_xlabel("Environment steps per agent")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M"))
fig.suptitle(a.title, x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2)
fig.savefig(a.out, dpi=170)
print("wrote", a.out)
