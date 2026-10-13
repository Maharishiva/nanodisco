"""Plot logged evidence, including failed runs; no smoothing or cherry-picking."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", type=Path)
p.add_argument("--out", type=Path, default=Path("results/curves.png"))
p.add_argument("--title", default="CPU evidence", help="Recorded device/run label")
a = p.parse_args()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
for folder in a.runs:
    info = json.loads((folder / "summary.json").read_text())
    c = info["config"]
    label = f"Q={int(c['q'])}, next={int(c['next_policy'])}"
    meta = np.load(folder / "meta_curve.npz")
    if len(meta["step"]):
        axes[0].plot(meta["step"], meta["sample_return"], label=label)
    for ax, env in zip(axes[1:], ("catch", "reach")):
        data = np.load(folder / f"{env}_curves.npz")
        curves = np.c_[data["before"], data["learned"]]
        (line,) = ax.plot(curves.mean(0), label=label)
        ax.fill_between(range(curves.shape[1]), curves.min(0), curves.max(0), alpha=0.15, color=line.get_color())
        control = np.c_[data["before"], data["untrained"]].mean(0)
        ax.plot(control, "--", alpha=0.5, color=line.get_color())
for ax, title, xlabel in zip(
    axes,
    ("Catch meta-training", "Catch: fresh agents", "Reach: frozen-rule transfer"),
    ("Meta-update", "Agent update", "Agent update"),
):
    ax.set(title=title, xlabel=xlabel, ylabel="Expected episode return")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Sampled validation episode return")
axes[1].legend(fontsize=8)
fig.suptitle(a.title + " · solid: learned rule · dashed: untrained rule · band: seed range", fontsize=10)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=160)
print(a.out)
