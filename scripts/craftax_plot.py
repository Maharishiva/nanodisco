"""Craftax-Classic with frozen rules: training return over steps and the final policy's achievements.

usage: python scripts/craftax_plot.py OUT.png RUN.npz [RUN.npz ...]  (npz files from scripts/rule_craftax.py)
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
COLORS = ("#e08a1e", "#2a78d6", "#9b9a96", "#7a3fb8")
LABEL = {"arc_ent0.2": "rule trained on ARC games", "catch_ent0.2": "rule trained on Catch",
         "minatar_ent0.1": "rule trained on MinAtar games", "v2_catch": "v2 rule (Catch)"}
ACHIEVEMENTS = ("collect wood", "place table", "eat cow", "collect sapling", "collect drink", "make wood pickaxe",
                "make wood sword", "place plant", "defeat zombie", "collect stone", "place stone", "eat plant",
                "defeat skeleton", "make stone pickaxe", "make stone sword", "wake up", "place furnace",
                "collect coal", "collect iron", "collect diamond", "make iron pickaxe", "make iron sword")
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 9.5})
runs = []
for path in sys.argv[2:]:
    z = np.load(path)
    for v, name in enumerate(z["names"]):
        key = str(name).split()[1]
        runs.append(dict(label=LABEL.get(key, key), log=z["log"][:, v].sum(1), per=int(z["steps_per_iter"]),
                         ret=z["eval_return"][v].ravel(), ach=z["eval_achievements"][v].reshape(-1, 22).mean(0)))
fig, (cur, ach) = plt.subplots(1, 2, figsize=(16, 6.5), width_ratios=(1, 1.5))
for c, r in zip(COLORS, runs):
    bins = np.array_split(np.arange(len(r["log"])), 20)  # 50k steps per point
    ret = [r["log"][b, 0].sum() / max(1, r["log"][b, 1].sum()) for b in bins]
    cur.plot([(b[-1] + 1) * r["per"] / 1e6 for b in bins], ret, color=c, lw=2, label=r["label"])
    cur.scatter(1.0, r["ret"].mean(), color=c, s=40, zorder=3, edgecolor="white")
for value, name in ((8.3, "our PPO, IMPALA + GRU: 8.3"), (6.6, "Disco103 (paper): 6.6"), (4.2, "PPO (paper): 4.2")):
    cur.axhline(value, color=MUTED, ls=":", lw=1.2)
    cur.text(0.02, value + 0.12, name, color=MUTED, fontsize=9)
cur.set_xlim(0, 1.03)
cur.set_ylim(0, 9.5)
cur.set_xlabel("Env steps (M)")
cur.set_ylabel("Return (training episodes; dot: final policy, 256 episodes)")
cur.set_title("Return, 1 agent per rule, reference lines at 1M steps", loc="left")
cur.legend(loc="center right", bbox_to_anchor=(1, 0.565), frameon=False)
w = 0.8 / max(1, len(runs))
for i, (c, r) in enumerate(zip(COLORS, runs)):
    ach.bar(np.arange(22) + (i - (len(runs) - 1) / 2) * w, 100 * r["ach"], width=w * 0.95, color=c, label=r["label"])
ach.set_xticks(range(22), ACHIEVEMENTS, rotation=60, ha="right", fontsize=9)
ach.set_ylabel("Success rate (%)")
ach.set_title("Achievements of the final policy (256 episodes)", loc="left")
ach.legend(loc="upper right", frameon=False)
for ax in (cur, ach):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Craftax-Classic pixels, 1M steps: frozen learned rules train an IMPALA + GRU agent (400k replay)",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(sys.argv[1], dpi=160)
for r in runs:
    top = np.argsort(-r["ach"])[:4]
    print(f"{r['label']:32s} final {r['ret'].mean():.2f}  " + ", ".join(f"{ACHIEVEMENTS[i]} {100 * r['ach'][i]:.0f}%"
                                                                     for i in top))
print("wrote", sys.argv[1])
