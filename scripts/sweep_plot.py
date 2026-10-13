"""What mattered for MinAtar Breakout: the meta-entropy coefficient per family, then the other meta-training knobs.

usage: python scripts/sweep_plot.py NIGHT_DIR [--out OUT.png] [--ref NPZ] [--ppo NPZ]
A rule is a NIGHT_DIR/<run>/ with q/summary.json and breakout/breakout.npz (scripts/rule_paper.py, one variant per eval
lr); runs without breakout.npz yet are skipped. Score: per-agent return over the last 10% of updates, at the rule's
best eval lr. Left: return vs META_ENT (0.01 when not overridden) for the runs of each family (env, agent core, rule
core) whose settings match the family's first run in all but META_ENT. Right: one knob changed at a time from n01
(MinAtar, MLP agent + attention rule, META_ENT 0.05).
--ref: the v2 headline rule (default results/speed_20261006/base_unroll4.npz, lr 5e-4 only).
--ppo: PPO on Breakout, the best of its lrs drawn as a line (default results/bench_20261005/breakout/ppo.npz).
Writes NIGHT_DIR/sweep.png by default and prints one line per rule.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#ebeae6", "#b9b7b0"
FAMILIES = {  # (env, agent core, rule core) -> label, color, marker; the 4 colors pass a CVD check
    ("catch", "gru", "gru"): ("Catch, GRU agent + GRU rule (a*)", "#2a78d6", "o"),
    ("catch", "mlp", "attn"): ("Catch, MLP agent + attn rule (c*)", "#e08a1e", "s"),
    ("minatar", "gru", "gru"): ("MinAtar, GRU agent + GRU rule", "#1baf7a", "o"),
    ("minatar", "mlp", "attn"): ("MinAtar, MLP agent + attn rule", "#d64545", "s"),
}
OTHER = "#8a4fd8"  # families without a line, e.g. the GDN-2 agent
KNOBS = (("n01", "base\nn01"), ("n07", "2nd\nmeta-seed\nn07"), ("n08", "1600\nmeta-steps\nn08"),
         ("n09", "meta lr\n3e-4\nn09"), ("n10", "no\nreplay\nn10"), ("n14", "GDN-2\nagent\nn14"),
         ("n11", "GRU agent\n+ GRU rule\nn11"), ("n13", "GRU + GRU\nmeta-seed 1\nn13"))
SAME = ("seed", "steps", "agents", "inner", "replay", "meta_lr", "batch", "buffer", "lifetimes", "games", "rollout")
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 9})
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("night", type=Path)
p.add_argument("--out", type=Path)
p.add_argument("--ref", type=Path, default=ROOT / "results/speed_20261006/base_unroll4.npz")
p.add_argument("--ppo", type=Path, default=ROOT / "results/bench_20261005/breakout/ppo.npz")
a = p.parse_args()

def sci(x):  # 0.0005 -> 5e-4
    mant, exp = f"{float(x):e}".split("e")
    return f"{float(mant):g}e{int(exp)}"

def score(path):  # per-agent last-10% return at the best eval lr, that lr, steps per agent, every eval lr
    z = np.load(path)
    log = z["log"]
    tail = log[-max(1, len(log) // 10):].sum(0)
    final = tail[..., 0] / np.maximum(tail[..., 1], 1)
    i = int(final.mean(-1).argmax())
    lrs = [sci(str(n).split("lr=")[-1]) for n in z["names"]]
    return final[i], lrs[i], len(log) * int(z["steps_per_iter"]), lrs

rules = {}
for f in sorted(a.night.glob("*/q/summary.json")):
    s, npz = json.loads(f.read_text()), f.parent.parent / "breakout" / "breakout.npz"
    if not npz.exists():
        print(f"{f.parent.parent.name:36s} no breakout.npz yet, skipped")
        continue
    ov = s.get("overrides") or {}
    final, lr, steps, lrs = score(npz)
    sig = tuple(str(s.get(k) or None) for k in SAME) + tuple(sorted((k, str(v)) for k, v in ov.items()
                                                           if k not in ("META_ENT", "UNROLL")))  # UNROLL: speed only
    fam = (s.get("env") or "catch", ov.get("AGENT_CORE", "gru").lower(), ov.get("RULE_CORE", "gru").lower())
    rules[f.parent.parent.name] = dict(fam=fam, ent=float(ov.get("META_ENT", 0.01)), final=final, lr=lr, sig=sig,
                                       steps=steps, lrs=lrs)
base = {}  # each family's first run sets the settings its line keeps
for name, x in rules.items():
    base.setdefault(x["fam"], x["sig"])
lines = {fam: sorted((x for x in rules.values() if x["fam"] == fam and x["sig"] == base[fam]), key=lambda x: x["ent"])
         for fam in FAMILIES if fam in base}
knobs = [(label, rules[n]) for prefix, label in KNOBS for n in rules if n.split("_")[0] == prefix]
ref, ref_lr = score(a.ref)[:2]
ref_label = f"v2 headline\nrule (eval\nlr {ref_lr} only)"
ppo = score(a.ppo)[:2] if a.ppo.exists() else None  # PPO at its best lr
lrs = sorted({lr for x in rules.values() for lr in x["lrs"]}, key=float)
for name, x in sorted(rules.items(), key=lambda kv: -kv[1]["final"].mean()):
    where = ("line" if any(x is y for y in lines.get(x["fam"], [])) else "") + \
        (" knobs" if any(x is k for _, k in knobs) else "")
    print(f"{name:36s} {'/'.join(x['fam']):20s} ent {x['ent']:<5g} lr {x['lr']:7s} {x['final'].mean():6.1f} "
          f"+- {x['final'].std() / np.sqrt(len(x['final'])):4.1f}  {where.strip() or 'not plotted'}")
fig, (ent_ax, knob_ax) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw=dict(width_ratios=(1, 1.2)))
rng = np.random.default_rng(0)
dodge = dict(zip(FAMILIES, (-0.03, -0.01, 0.01, 0.03)))  # in decades, so families at one META_ENT do not overlap
for fam, xs in lines.items():
    label, color, marker = FAMILIES[fam]
    ent = np.array([x["ent"] for x in xs]) * 10 ** dodge[fam]
    for e, x in zip(ent, xs):
        ent_ax.scatter(e * 10 ** rng.uniform(-0.008, 0.008, len(x["final"])), x["final"], s=12, color=color, alpha=0.45,
                       lw=0, zorder=2)
    ent_ax.plot(ent, [x["final"].mean() for x in xs], color=color, lw=2.2, marker=marker, ms=7, mec="white",
                label=label, zorder=3)
ent_ax.set_xscale("log")
ticks = sorted({x["ent"] for xs in lines.values() for x in xs})
ent_ax.xaxis.set_major_locator(FixedLocator(ticks))
ent_ax.xaxis.set_minor_locator(NullLocator())
ent_ax.set_xticklabels([f"{t:g}" for t in ticks])
ent_ax.set_xlabel("Meta-entropy coefficient META_ENT (log scale)")
ent_ax.set_title("Return vs meta-entropy, one line per family (dots: one agent each)", loc="left")
knob_ax.set_title("MinAtar MLP agent + attn rule at META_ENT 0.05: one change per bar\n"
                  "(dots: one agent each; dashed: base n01)", loc="left")
for i, (label, x) in enumerate(knobs + [(ref_label, dict(final=ref, fam=None))]):
    color = REF if x["fam"] is None else FAMILIES.get(x["fam"], (None, OTHER))[1]
    knob_ax.bar(i, x["final"].mean(), width=0.7, color=color, alpha=0.35)
    knob_ax.scatter(i + rng.uniform(-0.18, 0.18, len(x["final"])), x["final"], s=14,
                    color=MUTED if x["fam"] is None else INK, zorder=3)
    knob_ax.annotate(f"{x['final'].mean():.0f}", (i, x["final"].max()), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold", color=INK)
knob_ax.set_xticks(range(len(knobs) + 1), [k for k, _ in knobs] + [ref_label], fontsize=9)
if knobs and knobs[0][1]["fam"] == ("minatar", "mlp", "attn"):
    knob_ax.axhline(knobs[0][1]["final"].mean(), color=MUTED, lw=1, ls="--", zorder=1)
ent_ax.axhline(ref.mean(), color=MUTED, lw=1.2, ls="--", zorder=1)  # references: the v2 headline rule, PPO
ent_ax.annotate(f"v2 headline rule {ref.mean():.0f}", (1, ref.mean()), xycoords=("axes fraction", "data"),
                xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=9, color=MUTED)
if ppo:
    ent_ax.axhline(ppo[0].mean(), color=MUTED, lw=1.2, ls=":", zorder=1)
    ent_ax.annotate(f"PPO, best lr ({ppo[1]}) {ppo[0].mean():.0f}", (1, ppo[0].mean()),
                    xycoords=("axes fraction", "data"), xytext=(-2, 4), textcoords="offset points", ha="right",
                    fontsize=9, color=MUTED)
ent_ax.legend(loc="lower right", frameon=False)
top = max(x["final"].max() for x in rules.values()) * 1.1
for ax in (ent_ax, knob_ax):
    ax.set_ylim(0, top)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
ent_ax.set_ylabel(f"Return, last 10% of {max(x['steps'] for x in rules.values()) / 1e6:.0f}M steps")
fig.suptitle("What mattered on MinAtar Breakout: rules trained frozen, paper regime, 8 agents each, every rule at its "
             f"best eval lr ({' or '.join(lrs)})", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.5)
out = a.out or a.night / "sweep.png"
fig.savefig(out, dpi=170)
print("wrote", out)
