"""Does a three-term formula reproduce the learned rule? Train fresh agents with each and compare.

usage: python scripts/rule_symbolic.py OUT.npz [--env catch] [--updates 2000] [--agents 4]
Symbolic policy target, fitted to the value-input rule's updates on Catch (R2 0.64 on log(target/pi)):
    target logits_a = 0.57 log pi_a + 0.83 norm(Q_a - V) + 0.98 [a = a_t]
plus ablations that drop the taken-action boost, the flattening (log pi coefficient 1), or both.
norm() is over the whole batch, as in the rule's input. The Q head keeps its TD(lambda) loss; the symbolic
rule has no y/z targets. Both run in the discovery setting: 32 envs x 16 steps, one update per rollout.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import mini as m
import symbolic

VARIANTS = {
    "formula": (0.57, 0.83, 0.98),
    "no taken boost": (0.57, 0.83, 0.0),
    "no flattening": (1.0, 0.83, 0.98),
    "Q-advantage only": (1.0, 0.83, 0.0),
}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--env", default="catch", choices=("catch", "reach", "breakout"))
p.add_argument("--updates", type=int, default=2000)
p.add_argument("--agents", type=int, default=4)
p.add_argument("--envs", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
m.OBS, bench.ENV_RESET, bench.ENV_STEP = bench.make_env(a.env)
m.B, m.USE_Q = a.envs, True
eta = bench.load_rule(bench.RULES / "q" / "rule.npz")

def policy(th, h, s, first):
    return m.agent_step(th, h, s, first, "raw")

def update(kind, st, k, coef=None):
    th, opt, carry = st
    carry, d, log = bench.rollout(policy, th, carry, k)
    th, opt = m.inner_step(th, opt, eta, d, "raw")[:2] if kind == "rule" else symbolic.step(th, opt, d, "raw", coef)
    return (th, opt, carry), log

def init(k):
    ka, ke = jax.random.split(k)
    th = m.init_agent(ka)
    zeros = jax.tree.map(jnp.zeros_like, th)
    return th, (zeros, zeros, jnp.zeros(())), bench.init_carry(ke)

def run(kind, key, coef=None):  # same agents, envs and keys for every kind
    st = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.agents))
    def step(st, k):
        return jax.vmap(lambda s, kk: update(kind, s, kk, coef))(st, jax.random.split(k, a.agents))
    return jax.lax.scan(step, st, jax.random.split(jax.random.fold_in(key, 1), a.updates))[1]

key = jax.random.PRNGKey(3000 + a.seed)
coefs = jnp.asarray(list(VARIANTS.values()))
sym = np.asarray(jax.jit(lambda k: jax.vmap(lambda c: run("symbolic", k, c))(coefs))(key))
logs = {"rule": np.asarray(jax.jit(lambda k: run("rule", k))(key))} | dict(zip(VARIANTS, sym))
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, steps_per_update=m.B * m.ROLL, **logs)
w = max(1, a.updates // 10)
for kind, log in logs.items():
    r = log[: a.updates // w * w].reshape(-1, w, *log.shape[1:]).sum(1)
    print(
        f"{a.env:8s} {kind:17s}", " ".join(f"{x:+.2f}" for x in (r[..., 0].sum(-1) / np.maximum(r[..., 1].sum(-1), 1)))
    )
