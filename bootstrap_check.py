"""Frozen-rule intervention: replace next-state Y/Z by uniform predictions.

Connectivity alone does not demonstrate learned bootstrapping. This diagnostic
measures whether learned next-prediction inputs help held-out agent learning.
It cannot prove the predictions implement a Bellman equation.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import disco as d
from evaluate import lifetime

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, default=Path("results/neither"))
p.add_argument("--seeds", type=int, default=4)
p.add_argument("--inner", type=int, default=60)
a = p.parse_args()
c = json.loads((a.run / "summary.json").read_text())["config"]
eta = {k: jnp.asarray(v) for k, v in np.load(a.run / "rule.npz").items()}
original = d.features

def without_next_predictions(out, nxt, ema, ema_next, data):
    nxt, ema_next = dict(nxt), dict(ema_next)
    for o in (nxt, ema_next):
        for k in ("y", "z"):
            o[k] = jnp.zeros_like(o[k])  # Uniform after softmax; all else identical.
    return original(out, nxt, ema, ema_next, data)

summary = {}
for env in ("catch", "reach"):
    curves = {}
    for label, fn in (("intact", original), ("uniform_next_yz", without_next_predictions)):
        d.features = fn
        evaluate = jax.jit(lambda k: lifetime(eta, k, env, c["q"], c["next_policy"], a.inner, c["batch"], c["lr"])[1])
        curves[label] = np.stack([np.asarray(evaluate(jax.random.PRNGKey(10000 + s))) for s in range(a.seeds)])
    np.savez(a.run / f"{env}_bootstrap_intervention.npz", **curves)
    summary[env] = {
        k: dict(final=float(v[:, -1].mean()), mean_curve=float(v.mean()), seed_finals=v[:, -1].tolist())
        for k, v in curves.items()
    }
d.features = original
(a.run / "bootstrap_intervention.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
