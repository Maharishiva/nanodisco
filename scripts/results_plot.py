"""Main results: each rule on the 8 ARC holdout games and on MinAtar Breakout, grouped by the games it was trained on.

usage: python scripts/results_plot.py NIGHT_DIR [OUT.png]
NIGHT_DIR: synced night outputs (results/night_20261006). Rows without data yet are skipped.
ARC: levels per playthrough over the last 20% of 2M steps, one agent per game. Breakout: return over the last 10% of
5M steps, 8 agents, best of eval lr 5e-4 and 1e-3.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ebeae6"
COLOR = {"ARC games": "#e08a1e", "Catch": "#2a78d6", "MinAtar games": "#9b9a96", "control": "#d0cfca"}
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 9.5})
night = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else night / "results.png"
REPO = Path(__file__).resolve().parents[1]
# label, trained on, Breakout npz (night-relative or repo-relative), ARC file and arm
RULES = (
    ("GRU agent, GRU rule, meta-entropy 0.2", "ARC games", "r1_arc_gru_gru_ent0.2", ("r1_arc_gru_gru_ent0.2/holdout",
                                                                                     "r1_arc_gru_gru_ent0.2/q")),
    ("GRU agent, attention rule, meta-entropy 0.1", "ARC games", "r2_arc_gru_attn_ent0.1",
     ("r2_arc_gru_attn_ent0.1/holdout", "r2_arc_gru_attn_ent0.1/q")),
    ("GRU agent, GRU rule, meta-entropy 0.05", "ARC games", "r3_arc_gru_gru_ent0.05",
     ("r3_arc_gru_gru_ent0.05/holdout", "r3_arc_gru_gru_ent0.05/q")),
    ("GDN-2 agent, attention rule, meta-entropy 0.1", "ARC games", "r4_arc_gdn_attn_ent0.1",
     ("r4_arc_gdn_attn_ent0.1/holdout", "r4_arc_gdn_attn_ent0.1/q")),
    ("MLP agent, attention rule, meta-entropy 0.1", "ARC games", "r5_arc_mlp_attn_ent0.1",
     ("r5_arc_mlp_attn_ent0.1/holdout", "r5_arc_mlp_attn_ent0.1/q")),
    ("GRU agent, GRU rule, meta-entropy 0.2", "Catch", "a07_metaent0.2", ("r1_arc_gru_gru_ent0.2/holdout",
                                                                           "a07_metaent0.2/q")),
    ("v2 headline rule (old code)", "Catch", "results/speed_20261006/base_unroll4.npz",
     ("arc_holdout_gru", "catch_gpu_20261004/q")),
    ("v2 setup, new code", "Catch", "v2_recipe_catch", ("v2_recipe_catch/holdout", "v2_recipe_catch/q")),
    ("v3, meta-entropy 0.01 (first run)", "Catch", "results/entropy_20261006/v3_rule.npz",
     ("arc_holdout_gru", "v3_catch_20261006/q")),
    ("v2 setup, new code", "MinAtar games", "v2_recipe_minatar", ("v2_recipe_minatar/holdout", "v2_recipe_minatar/q")),
    ("GRU agent, GRU rule, meta-entropy 0.1", "MinAtar games", "n12_minatar_ent0.1_gru",
     ("arc_holdout_gru2", "n12_minatar_ent0.1_gru/q")),
    ("GRU agent, GRU rule, meta-entropy 0.05", "MinAtar games", "n11_minatar_ent0.05_gru",
     ("arc_holdout_gru", "n11_minatar_ent0.05_gru/q")),
    ("MLP agent, attention rule, meta-entropy 0.1", "MinAtar games", "n05_minatar_ent0.1",
     ("arc_holdout_mlpattn", "n05_minatar_ent0.1/q")),
    ("GDN-2 agent, attention rule, meta-entropy 0.05", "MinAtar games", "n14_minatar_ent0.05_gdnattn",
     ("arc_holdout_gdnattn", "n14_minatar_ent0.05_gdnattn/q")),
    ("random policy", "control", None, ("r1_arc_gru_gru_ent0.2/holdout", "random")),
    ("untrained rule", "control", None, ("r1_arc_gru_gru_ent0.2/holdout", "untrained")),
)

def arc(file, arm):  # per-game levels per playthrough, last 20% of updates; games
    path = night / file / "arc_eval.npz"
    if not path.exists():
        return None, None
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    if arm not in names:
        return None, None
    log = z["log"][names.index(arm)]
    tail = log[-len(log) // 5:].sum(0)
    return tail[:, 0] / np.maximum(tail[:, 1], 1), [str(g) for g in z["games"]]

def breakout(path):  # per-agent return, last 10% of updates, at the best eval lr
    if path is None:
        return None
    p = REPO / path if path.startswith("results/") else night / path / "breakout" / "breakout.npz"
    if not p.exists():
        return None
    log = np.load(p)["log"]
    tail = log[-max(1, len(log) // 10):].sum(0)
    finals = tail[..., 0] / np.maximum(tail[..., 1], 1)  # [variants, seeds]
    return finals[finals.mean(1).argmax()]

rows = []
for label, env, bpath, (afile, arm) in RULES:
    lpp, games = arc(afile, arm)
    ret = breakout(bpath)
    if lpp is not None or ret is not None:
        rows.append(dict(label=f"{env}: {label}" if env != "control" else label, env=env, lpp=lpp, games=games,
                         ret=ret))
for r in rows:
    a = "-" if r["lpp"] is None else f"{r['lpp'].mean():.2f}"
    b = "-" if r["ret"] is None else f"{r['ret'].mean():.1f}"
    print(f"{r['label']:62s} ARC {a:>5s}  Breakout {b:>5s}")
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), width_ratios=(1, 1))
arc_ax, brk_ax, game_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
y = np.arange(len(rows))[::-1]
rng = np.random.default_rng(0)
for yi, r in zip(y, rows):
    c = COLOR[r["env"]]
    if r["lpp"] is not None:
        arc_ax.barh(yi, r["lpp"].mean(), color=c, height=0.7)
        arc_ax.scatter(r["lpp"], yi + rng.uniform(-0.2, 0.2, len(r["lpp"])), s=10, color=INK, zorder=3)
        arc_ax.text(2.95, yi, f"{r['lpp'].mean():.2f}", va="center", ha="right", fontweight="bold")
    if r["ret"] is not None:
        brk_ax.barh(yi, r["ret"].mean(), color=c, height=0.7)
        brk_ax.scatter(r["ret"], yi + rng.uniform(-0.2, 0.2, len(r["ret"])), s=10, color=INK, zorder=3)
        brk_ax.text(98, yi, f"{r['ret'].mean():.0f}", va="center", ha="right", fontweight="bold")
arc_ax.set_yticks(y, [r["label"] for r in rows], fontsize=9.5)
brk_ax.set_yticks(y, [])
random = next((r["lpp"].mean() for r in rows if r["label"] == "random policy"), None)
if random is not None:
    arc_ax.axvline(random, color=MUTED, ls="--", lw=1)
brk_ax.axvline(33, color=MUTED, ls=":", lw=1.2)
brk_ax.text(33.5, y[0] + 0.55, "PPO 33", color=MUTED, fontsize=9)
arc_ax.set_xlim(0, 3)
brk_ax.set_xlim(0, 100)
arc_ax.set_title("ARC holdout, 8 unseen games (dots: games; dashed: random)", loc="left")
brk_ax.set_title("MinAtar Breakout (dots: one agent)", loc="left")
arc_ax.set_xlabel("Levels per playthrough, last 20% of 2M steps")
brk_ax.set_xlabel("Return, last 10% of 5M steps")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in COLOR]
fig.legend(handles, ["trained on ARC games", "trained on Catch",
                     "trained on 4 MinAtar games (Breakout-like: not a fair Breakout test)", "control"],
           loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=4, frameon=False)
show = [r for r in rows if r["lpp"] is not None and (r["env"] in ("ARC games", "control") or
                                                   r["label"].startswith("Catch: GRU agent"))][:5]
games = show[0]["games"] if show else []
w = 0.8 / max(1, len(show))
for i, r in enumerate(show):
    c = COLOR[r["env"]] if r["env"] != "control" else (MUTED if r["label"] == "random policy" else "#d0cfca")
    game_ax.bar(np.arange(len(games)) + (i - (len(show) - 1) / 2) * w, r["lpp"], width=w * 0.95, color=c,
                label=r["label"], alpha=1.0 if i == 0 else 0.75)
game_ax.set_xticks(range(len(games)), games)
game_ax.set_ylabel("Levels per playthrough")
game_ax.set_title("Per holdout game", loc="left")
game_ax.legend(loc="upper right", frameon=False, ncol=2)
for ax in (arc_ax, brk_ax, game_ax):
    ax.grid(axis="y" if ax is game_ax else "x", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Learned rules, frozen, train fresh agents: ARC holdout games and MinAtar Breakout", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5, w_pad=1.5)
fig.savefig(out, dpi=160)
print("wrote", out)
