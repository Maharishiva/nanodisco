"""Training return of persistent agents on Catch under three update rules.

usage: python scripts/catch_run.py OUT STEPS [SEED]
learned: meta-trained rule (mini.train); frozen: the same random rule, never
meta-trained; a2c: hand-written actor-critic using the agent's own Q-head
advantages, same agents, Adam, rollouts and update count.
"""

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "history"))
import mini_v2 as m  # written for the on-policy mini v2

def a2c(key, steps):
    def loss(th, d):
        full = m.unroll(th, d, "catch")
        qs, out = m.q_stats(full, d), jax.tree.map(lambda x: x[:-1], full)
        logp, qa = m.pick(m.lsm(out["pi"]), d["a"])[..., 0], m.pick(out["q"], d["a"])[..., 0]
        return (-logp * m.norm(qs["adv"]) + 0.5 * (qa - qs["g"]) ** 2).mean() - 1e-2 * m.entropy(out["pi"])
    def one_agent(st, k):
        th, opt, phi, copt, carry = st
        kt, kv = jax.random.split(k)
        carry, train = jax.lax.scan(lambda c, k: m.collect(th, c, k, "catch"), carry, jax.random.split(kt, m.INNER))
        carry, valid = m.collect(th, carry, kv, "catch", 2 * m.ROLL)
        th, opt = jax.lax.scan(lambda c, d: (m.adam(*c, jax.grad(loss)(c[0], d)), None), (th, opt), train)[0]
        return (th, opt, phi, copt, carry), m.episode_return(train, valid)
    def step(agents, k):
        agents, ret = jax.vmap(one_agent)(agents, jax.random.split(k, m.AGENTS))
        return agents, ret.mean()
    _, ka, ks = jax.random.split(key, 3)
    return jax.lax.scan(
        step, jax.vmap(m.init_agent_state)(jax.random.split(ka, m.AGENTS)), jax.random.split(ks, steps)
    )[1]

out, steps = Path(sys.argv[1]), int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=True)
key = jax.random.PRNGKey(seed)
arms = dict(
    learned=lambda k: m.train(k, steps)[2]["train_return"],
    frozen=lambda k: m.train(k, steps, 0.0)[2]["train_return"],
    a2c=lambda k: a2c(k, steps),
)
summary = dict(
    steps=steps,
    seed=seed,
    agents=m.AGENTS,
    envs_per_agent=m.B,
    rollout=m.ROLL,
    inner=m.INNER,
    devices=[str(d) for d in jax.devices()],
    arms={},
)
for name, fn in arms.items():
    start = time.monotonic()
    curve = np.asarray(jax.block_until_ready(jax.jit(fn)(key)))
    np.save(out / f"{name}_train_return.npy", curve)
    window = max(1, steps // 20)
    means = [float(curve[i : i + window].mean()) for i in range(0, steps, window)]
    summary["arms"][name] = dict(seconds=time.monotonic() - start, windowed=means, last_window=means[-1])
    print(name, f"{time.monotonic() - start:.0f}s", " ".join(f"{x:+.2f}" for x in means), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
