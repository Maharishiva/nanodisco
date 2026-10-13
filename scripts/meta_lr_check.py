"""Issue 3 check: does meta_lr_scale scale the meta-update? One meta-step from the same state at scale 1 and 0.1."""

import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mini as m

key = jax.random.PRNGKey(0)
eta0 = m.init_rule(jax.random.split(key, 3)[0])  # the rule train() starts from
def size(scale):
    eta = jax.jit(lambda k: m.train(k, 1, meta_lr_scale=scale)[0])(key)
    return np.sqrt(sum(float(((x - y) ** 2).sum()) for x, y in zip(jax.tree.leaves(eta), jax.tree.leaves(eta0))))
full, tenth = size(1.0), size(0.1)
print(f"meta-update norm, scale 1: {full:.5f}; scale 0.1: {tenth:.5f}; ratio {tenth / full:.3f} (0.1 if it works)")
