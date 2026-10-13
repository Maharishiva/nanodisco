"""How much Breakout return depends on the meta-training seed: every Catch-trained rule we have, trained frozen.

usage: python scripts/rules_spread_plot.py BENCH_DIR [OUT.png]
Reads breakout/outer_rules.npz (outer_check pre/post rules), breakout/headline_check.npz (the headline rule and
4 new mini.train rules), breakout/v3_rule.npz (the mini v3 rule) and breakout/symbolic_paper.npz.
All: paper regime, lr 5e-4, 8 agents per rule, 5M steps.
Third panel: every PPO learning rate, on-policy (ppo.npz, 4 epochs x 4 minibatches) and in the paper regime
(ppo_paper.npz, 1 x 4), 8 agents each.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
V3 = "#8a4fd8"
GROUPS = (  # (file, name prefix, legend label, tick label, color); all but v3 are mini v2 rules
    ("outer_rules", "pre/", "v2 outer_check, validation data before adaptation", "before", "#2a78d6"),
    ("outer_rules", "post/", "v2 outer_check, validation data after adaptation", "after", "#c03a8f"),
    ("headline_check", "mini_train", "v2 mini.train (headline's own script)", "mini.train", "#1baf7a"),
    ("headline_check", "headline", "v2 headline rule", "headline", INK),
    ("v3_rule", "lr=0.0005", "v3 rule (replay meta-training, 1 meta-seed)", "v3", V3),
)
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

def tint(color, t):  # mix toward white: t=0 full color
    return tuple(c + (1 - c) * t for c in to_rgb(color))

def sci(x):  # 0.00025 -> 2.5e-4
    mant, exp = f"{float(x):e}".split("e")
    return f"{float(mant):g}e{int(exp)}"

def load(path):  # names, windowed curves [variants, POINTS] (mean over agents), last-10% return [variants, agents]
    z = np.load(path)
    log = z["log"]
    w = max(1, len(log) // POINTS)
    n = len(log) // w * w
    win = log[:n].reshape(n // w, w, *log.shape[1:]).sum(1)
    curves = (win[..., 0] / np.maximum(win[..., 1], 1)).mean(-1).T
    tail = log[-max(1, len(log) // 10) :].sum(0)
    final = tail[..., 0] / np.maximum(tail[..., 1], 1)
    steps = (np.arange(n // w) + 1) * w * int(z["steps_per_iter"])
    return [str(x).split()[1] for x in z["names"]], curves, final, steps

bench = Path(sys.argv[1]) / "breakout"
fig, (curve_ax, dot_ax, ppo_ax) = plt.subplots(1, 3, figsize=(21, 5.8), gridspec_kw=dict(width_ratios=(1.1, 1.5, 1)))
rng = np.random.default_rng(0)
means, x0, ticks = [], 0.0, []

def column(x, agents, color):  # one rule: a small dot per agent, a bar at their mean
    dot_ax.scatter(x + rng.uniform(-0.18, 0.18, len(agents)), agents, s=16, color=color, alpha=0.55, lw=0, zorder=2)
    dot_ax.plot([x - 0.32, x + 0.32], [agents.mean()] * 2, color=color, lw=3, solid_capstyle="butt", zorder=3)

for g, (arm, prefix, label, tick, color) in enumerate(GROUPS):
    names, curves, final, steps = load(bench / f"{arm}.npz")
    mine = [i for i, n in enumerate(names) if n.startswith(prefix)]
    single = len(mine) == 1  # the headline and v3 rules: bold
    for i in mine:
        curve_ax.plot(steps, curves[i], color=color, lw=2.4 if single else 1.4, ls="--" if color == INK else "-",
                      alpha=1 if single else 0.8, zorder=3 if single else 2)
    if tick == "v3":
        curve_ax.annotate("v3", (steps[-1], curves[mine[0]][-1]), xytext=(5, 0), textcoords="offset points",
                          va="center", fontsize=10.5, fontweight="bold", color=color)
    for j, i in enumerate(mine):
        column(x0 + j, final[i], color)
        if tick != "v3":
            means.append(final[i].mean())
    ticks.append(x0 + (len(mine) - 1) / 2)
    x0 += len(mine) + (1.4 if len(mine) > 1 else 1.9)
names, curves, final, steps = load(bench / "symbolic_paper.npz")
i = names.index("lr=0.0005")
curve_ax.plot(steps, curves[i], color="#eb6834", lw=2.2, ls=":", zorder=3)
column(x0, final[i], "#eb6834")
ticks.append(x0)
x0 += 2.4
names, curves, final, steps = load(bench / "ppo.npz")  # on-policy: 128 envs x 16 steps, 4 epochs x 4 minibatches
ppo = []
for i, n in enumerate(names):
    lr = n.split("=")[1]
    if lr == "0.0005":
        ppo.append(ppo_ax.plot(steps, curves[i], color=INK, lw=2.8, zorder=4, label=f"on-policy, lr {sci(lr)}")[0])
    else:
        t = (0.62, 0.45, 0.28)[len([x for x in ppo if x.get_lw() < 2])]
        line = ppo_ax.plot(steps, curves[i], color=tint(MUTED, t), lw=1.5, zorder=3, label=f"on-policy, lr {sci(lr)}")
        ppo.append(line[0])
    column(x0 + i, final[i], ppo[-1].get_color())  # one column per learning rate
ticks.append(x0 + (len(names) - 1) / 2)
x0 += len(names) + 1.4
names, curves, final, steps = load(bench / "ppo_paper.npz")  # paper regime, 1 epoch x 4 minibatches
for i in range(len(names)):
    line = ppo_ax.plot(steps, curves[i], color=MUTED, lw=1.5, ls=":", zorder=3)[0]
    column(x0 + i, final[i], MUTED)
line.set_label("paper regime (1 x 4), all 4 lrs")
ticks.append(x0 + (len(names) - 1) / 2)
x0 += len(names) - 1
dot_ax.axhline(np.mean(means), color=MUTED, lw=1, ls="--", zorder=1)
dot_ax.set_xticks(ticks, [g[3] for g in GROUPS] + ["symbolic", "PPO\non-policy", "PPO\npaper regime"])
dot_ax.set_xlim(-0.6, x0 + 0.6)
dot_ax.set_title(f"Last 10%. Dot: 1 agent. Bar: mean of 8 agents (1 rule, or 1 PPO lr:\n"
                 f"2.5e-4, 5e-4, 1e-3, 2.5e-3 from left). "
                 f"Dashed: mean of the {len(means)} learned v2 rules ({np.mean(means):.0f}).", loc="left")
ppo_ax.axhline(np.mean(means), color=MUTED, lw=1, ls="--", zorder=1, label="mean of the 13 learned v2 rules")
ppo_ax.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=2.4)
ppo_ax.set_title("PPO, every learning rate (mean of 8 agents)", loc="left")
for ax in (curve_ax, ppo_ax):
    ax.set_xlabel("Environment steps per agent")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
for ax in (curve_ax, dot_ax, ppo_ax):
    ax.set_ylim(0, 75)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
curve_ax.set_title("Rules, one line per rule (mean of 8 agents)", loc="left")
curve_ax.set_ylabel("Episode return")
handles = [Line2D([], [], color=c, lw=4 if c != INK else 2.4, ls="--" if c == INK else "-") for *_, c in GROUPS]
handles.append(Line2D([], [], color="#eb6834", lw=2.4, ls=":"))
labels = [g[2] for g in GROUPS] + ["Symbolic formula of the headline rule"]
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, 0.94),
           fontsize=10)
fig.suptitle("MinAtar Breakout: 14 rules meta-trained on Catch and trained frozen (paper regime, lr 5e-4), and PPO",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else bench.parent / "rules_spread.png"
fig.savefig(out, dpi=200)
print("wrote", out)
