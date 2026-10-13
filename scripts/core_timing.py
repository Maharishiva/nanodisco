"""ms per frozen-rule agent update in scripts/rule_paper.py's Breakout regime, for several mini settings at once.

usage: python scripts/core_timing.py [--arms gru/gru/1 mlp/attn/0/assoc ...] [--iters 200] [--rounds 3] [--seeds 8]
An arm is AGENT_CORE/RULE_CORE/UNROLL (0: full unroll), plus /assoc for associative-scan returns.
Each update is rule_paper.py's: collect 8 envs x 29 steps, store, sample 96 x 29 (90% replay), one inner_step, for
--seeds x 3 lrs agents in one vmap. Untrained rules: timing only. Arms are compiled first, then run in alternating
rounds, so load from other GPU jobs hits all of them alike; the table shows the best round.
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

ENVS, T, BATCH, REPLAY = 8, 29, 96, 0.9
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--arms", nargs="+", default=["gru/gru/1", "gru/gru/0", "mlp/attn/0/assoc"])
p.add_argument("--iters", type=int, default=200, help="updates per timed call")
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--cap", type=int, default=1000, help="buffer trajectories (rule_paper: 13793); only memory")
a = p.parse_args()
CAP = a.cap
m.OBS, bench.ENV_RESET, bench.ENV_STEP = bench.make_env("breakout")
m.B, m.ROLL, m.USE_Q, m.RETRACE, m.AGENT_LR = ENVS, T, True, True, 1.0
lrs = jnp.asarray([1e-4, 3e-4, 5e-4])

def setting(arm):
    agent, rule, unroll, *assoc = arm.split("/")
    m.AGENT_CORE, m.RULE_CORE, m.UNROLL, m.ASSOC = agent, rule, int(unroll) or True, assoc == ["assoc"]

def policy(th, h, s, first):
    return m.agent_step(th, h, s, first, "raw")

def init(k):
    ka, ke = jax.random.split(k)
    th = m.init_agent(ka)
    zeros = jax.tree.map(jnp.zeros_like, th)
    def z(shape, dt=jnp.float32):
        return jnp.zeros((CAP,) + shape, dt)
    buf = dict(s=z((T + 1, m.OBS), jnp.uint8), first=z((T + 1,)), a=z((T,), jnp.int32), r=z((T,)), done=z((T,)),
               mu=z((T, m.A)), h0=z((getattr(m, "memory_size", lambda: m.HID)(),)))  # HEAD mini: no memory_size
    return th, (zeros, zeros, jnp.zeros(())), bench.init_carry(ke), buf | dict(size=jnp.zeros((), jnp.int32))

def iteration(eta, lr, state, k):  # as rule_paper.iteration with learn=True
    th, opt, carry, buf = state
    kc, kr, kf, km = jax.random.split(k, 4)
    carry, d, log = bench.rollout(policy, th, carry, kc)
    traj = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in d.items()}
    traj["s"] = traj["s"].astype(jnp.uint8)
    i = (buf["size"] + jnp.arange(ENVS)) % CAP
    buf = {k: buf[k].at[i].set(v) for k, v in traj.items()} | dict(size=jnp.minimum(CAP, buf["size"] + ENVS))
    replay = jax.tree.map(lambda x: x[jax.random.randint(kr, (BATCH,), 0, buf["size"])], {k: buf[k] for k in traj})
    fresh = jax.tree.map(lambda x: x[jax.random.randint(kf, (BATCH,), 0, ENVS)], traj)
    use = jax.random.bernoulli(km, REPLAY, (BATCH,))
    batch = jax.tree.map(lambda r, f: jnp.where(use.reshape((-1,) + (1,) * (r.ndim - 1)), r, f), replay, fresh)
    batch = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in batch.items()}
    batch["s"] = batch["s"].astype(jnp.float32)
    new, opt = m.inner_step(th, opt, eta, batch, "raw")[:2]
    th = jax.tree.map(lambda p, q: p + lr * (q - p), th, new)
    return (th, opt, carry, buf), log

def make_run():  # a new function per arm: jit would reuse the first arm's trace for the same function
    def run(eta, state, key):
        def body(st, k):
            ks = jax.random.split(k, a.seeds)
            return jax.vmap(lambda lr, st: jax.vmap(lambda s, kk: iteration(eta, lr, s, kk))(st, ks))(lrs, st)
        return jax.lax.scan(body, state, jax.random.split(key, a.iters))
    return run

key, arms = jax.random.PRNGKey(0), {}
for arm in a.arms:
    setting(arm)
    eta = m.init_rule(key)
    state = jax.vmap(init)(jax.random.split(key, a.seeds))
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(lrs),) + x.shape), state)
    start = time.monotonic()
    compiled = jax.jit(make_run()).lower(eta, state, key).compile()
    arms[arm] = [compiled, eta, state, time.monotonic() - start, []]
    print(f"{arm:20s} compiled in {arms[arm][3]:.0f}s", flush=True)
for r in range(a.rounds + 1):  # round 0 warms up
    for arm, x in arms.items():
        start = time.monotonic()
        x[2], log = jax.block_until_ready(x[0](x[1], x[2], jax.random.fold_in(key, r)))
        if r:
            x[4].append((time.monotonic() - start) / a.iters * 1e3)
print(f"{'arm':20s} {'compile s':>9s} {'ms/update best':>14s} {'median':>7s}  ({a.seeds} seeds x 3 lrs per update)")
for arm, x in arms.items():
    print(f"{arm:20s} {x[3]:9.0f} {min(x[4]):14.2f} {np.median(x[4]):7.2f}", flush=True)
