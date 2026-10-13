"""Record what the frozen value-input rule sees and outputs while it trains fresh agents on Catch.

usage: python scripts/rule_probe.py OUT.npz [--updates 2000] [--every 20] [--agents 4]
Setting as in discovery: 32 envs x 16-step rollouts, one update per rollout, mini.inner_step. Every --every
updates it stores, for the rollout about to be learned from: observations, actions, rewards, done flags,
behaviour and current logits, the agent's y/z and Q outputs, the value inputs (v, Q(lambda) returns, advantages,
Q - v), the rule's per-step and per-action input features, and its targets (pi_hat, y_hat, z_hat).
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import mini as m

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--updates", type=int, default=2000)
p.add_argument("--every", type=int, default=20)
p.add_argument("--agents", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
m.OBS, bench.ENV_RESET, bench.ENV_STEP = bench.make_env("catch")
m.USE_Q = True
eta = bench.load_rule(bench.RULES / "q" / "rule.npz")

def policy(th, h, s, first):
    return m.agent_step(th, h, s, first, "raw")

def probe(th, d):  # everything the rule sees and produces for this rollout
    out = m.unroll(th, d, "raw")
    qs = m.q_stats(out, d)
    step_f, act_f = m.features(out, d, qs)
    pi_hat, y_hat, z_hat = m.rule(eta, step_f, act_f, d)
    keep = {k: d[k] for k in ("s", "first", "a", "r", "done", "mu")}
    return keep | dict(
        pi=out["pi"],
        y=out["y"],
        z=out["z"],
        q=out["q"],
        v=qs["v"],
        g=qs["g"],
        adv=qs["adv"],
        qv=qs["qv"],
        step_f=step_f,
        act_f=act_f,
        pi_hat=pi_hat,
        y_hat=y_hat,
        z_hat=z_hat,
    )

def update(st, k):
    th, opt, carry = st
    carry, d, log = bench.rollout(policy, th, carry, k)
    th, opt = m.inner_step(th, opt, eta, d, "raw")[:2]
    return (th, opt, carry), (d, log)

def block(st, k):  # probe the first rollout of the block, then learn from it and the rest
    k0, k1 = jax.random.split(k)
    th, opt, carry = st
    carry, d, log0 = bench.rollout(policy, th, carry, k0)
    rec = probe(th, d)
    th, opt = m.inner_step(th, opt, eta, d, "raw")[:2]
    (th, opt, carry), (_, logs) = jax.lax.scan(update, (th, opt, carry), jax.random.split(k1, a.every - 1))
    return (th, opt, carry), (rec, jnp.concatenate([log0[None], logs]))

def init(k):
    ka, ke = jax.random.split(k)
    th = m.init_agent(ka)
    zeros = jax.tree.map(jnp.zeros_like, th)
    return th, (zeros, zeros, jnp.zeros(())), bench.init_carry(ke)

def run(key):
    st = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.agents))
    def step(st, k):
        return jax.vmap(block)(st, jax.random.split(k, a.agents))
    return jax.lax.scan(step, st, jax.random.split(jax.random.fold_in(key, 1), a.updates // a.every))[1]

start = time.monotonic()
rec, log = jax.block_until_ready(jax.jit(run)(jax.random.PRNGKey(3000 + a.seed)))
log = np.asarray(log)  # [snapshots, agents, every, 2]
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    a.out,
    every=a.every,
    steps_per_update=m.B * m.ROLL,
    log=np.asarray(log),
    **{k: np.asarray(v) for k, v in rec.items()},
)
ret = log[..., 0].sum(-1) / np.maximum(log[..., 1].sum(-1), 1)  # [snapshots, agents]
print(
    f"{time.monotonic() - start:.0f}s; return per snapshot block (mean over agents):",
    " ".join(f"{x:+.2f}" for x in ret.mean(1)[:: max(1, len(ret) // 20)]),
)
