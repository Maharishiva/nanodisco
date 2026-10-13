"""How matmul precision changes one meta-gradient of mini.train (full v3 settings, buffer filled first).

usage: python scripts/precision_check.py
Prints cosine similarity and relative error against 'highest' (float32), and for scale, a float32
meta-gradient from other random data (sampling noise).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mini as m

key = jax.random.PRNGKey(0)
eta = m.init_rule(jax.random.fold_in(key, 1))
agent = m.init_agent_state(jax.random.fold_in(key, 2))
params, opt_state, critic, critic_opt_state, carry, buffer = agent
for i in range(m.BATCH // m.B):  # fill the buffer with one batch, so the agent learns from the first update
    carry, rollout = m.collect(params, carry, jax.random.fold_in(key, 10 + i), "catch")
    buffer = m.store(buffer, rollout)
agent = (params, opt_state, critic, critic_opt_state, carry, buffer)
def flat(tree):
    return jnp.concatenate([x.ravel() for x in jax.tree.leaves(tree)])
grads = {}
for precision in ("highest", "tensorfloat32", "bfloat16"):
    with jax.default_matmul_precision(precision):
        grads[precision] = flat(jax.jit(jax.grad(lambda e: m.meta_loss(e, agent, jax.random.fold_in(key, 3))[0]))(eta))
with jax.default_matmul_precision("highest"):  # same agent, other data: the meta-gradient's sampling noise
    other = jax.jit(jax.grad(lambda e: m.meta_loss(e, agent, jax.random.fold_in(key, 4))[0]))(eta)
    grads["highest, other key"] = flat(other)
ref = grads["highest"]
for precision, g in grads.items():
    cosine = float(g @ ref / (jnp.linalg.norm(g) * jnp.linalg.norm(ref)))
    error = float(jnp.linalg.norm(g - ref) / jnp.linalg.norm(ref))
    print(f"{precision:14s} cosine {cosine:.4f}  relative error {error:.4f}", flush=True)
