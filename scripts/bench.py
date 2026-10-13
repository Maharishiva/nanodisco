"""Frozen discovered rule vs A2C and PPO, with matched networks, data and env steps.

usage: python scripts/bench.py OUT ENV ARM [--steps N] [--seeds S] [--envs E] [--actions N]
ENV: catch | reach (mini.py) | breakout (MinAtar, scripts/breakout.py)
     | asterix | freeway | seaquest | space_invaders (MinAtar, scripts/minatar.py, minimal action sets)
ARM: rule_q   the Catch-trained rule with value inputs, frozen, plus the untrained rule it started from
     rule_noq the Catch-trained rule without value inputs (staggered lifetimes 200), frozen
     symbolic the three-term formula of scripts/symbolic.py, in place of the value-input rule
     ppo      PPO, ported from purejaxrl's recurrent PPO: GAE, clipped policy and value loss,
              advantage normalization per minibatch, 4 epochs x 4 env-axis minibatches, learning-rate sweep
     a2c      the same code with 1 epoch and 1 minibatch (ratio 1, so clipping never binds)
Every arm: 2x256 MLP + 256 GRU torso, --envs envs (32 as in meta-training), 16-step rollouts, one update
phase per rollout, the same seeds, environments and initial torso weights. Logs the return of finished
training episodes.
--net impala (ppo, a2c; MinAtar games): the agent of our Craftax PPO in place of the MLP torso (mini's AGENT_CORE
impala: IMPALA CNN on the 10x10 grid, fc, GRU on [fc, previous action], a layer after it); the rest is unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import breakout as bk
import minatar as mn
import mini as m
import symbolic

RULES = ROOT / "results" / "catch_gpu_20261004"
LRS = (2.5e-4, 5e-4, 1e-3, 2.5e-3)
CLIP, VF, ENT, MAX_NORM = 0.2, 0.5, 0.01, 0.5
mini_obs, m.obs = m.obs, lambda s, env: s  # rollouts store observations, so every agent reads them directly
m.B, m.ROLL, m.RETRACE = 32, 16, False  # mini v2's setting, in which the saved rules were meta-trained and benchmarked

# ---------------------------------------------------------------- environments with auto-reset
def make_env(name):
    if name == "breakout":
        def reset(k, n):
            s = jax.vmap(bk.reset)(jax.random.split(k, n))
            return s, jax.vmap(bk.obs)(s)
        def step(k, s, a):
            s, r, done = jax.vmap(bk.step)(s, a)
            s = jax.tree.map(
                lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), reset(k, len(a))[0], s
            )
            return s, jax.vmap(bk.obs)(s), r, done
        return bk.OBS, reset, step
    if name in mn.GAMES:  # the agent picks among the game's minimal action set, as in breakout; sets m.A
        g = mn.GAMES.index(name)
        acts = jnp.asarray(np.flatnonzero(np.asarray(mn.VALID)[g]))
        m.A = len(acts)
        def reset(k, n):
            s = mn.reset(k, g, n)
            return s, mn.obs(s, g)
        def step(k, s, a):
            s, r, done = mn.step(k, s, acts[a], g)
            return s, mn.obs(s, g), r, done
        return mn.OBS, reset, step
    def reset(k, n):
        s = m.reset(k, name, n)
        return s, mini_obs(s, name)
    def step(k, s, a):
        s, r, done = m.env_step(s, a, name)
        s = jnp.where(done[:, None], m.reset(k, name, len(a)), s)
        return s, mini_obs(s, name), r, done
    return m.OBS, reset, step

def use_impala():  # the agent of our Craftax PPO on MinAtar grids: the grid as a frame (x255), then the previous action
    global ENV_RESET, ENV_STEP
    m.AGENT_CORE, m.PIXELS, grid_reset, grid_step = "impala", (10, 10, m.OBS // 100), ENV_RESET, ENV_STEP
    m.OBS += 1
    def ENV_RESET(k, n):
        s, o = grid_reset(k, n)
        return s, jnp.concatenate([255.0 * o, jnp.zeros((n, 1))], -1)
    def ENV_STEP(k, s, act):  # the previous action is 0 after a reset
        s, o, r, done = grid_step(k, s, act)
        return s, jnp.concatenate([255.0 * o, jnp.where(done, 0, act)[:, None].astype(o.dtype)], -1), r, done

def init_carry(k):  # env state, observation, memory, episode-start flag, running episode return
    s, o = ENV_RESET(k, m.B)
    return s, o, jnp.zeros((m.B, m.memory_size())), jnp.ones(m.B), jnp.zeros(m.B)

def rollout(policy, th, carry, key):  # same record as mini.rollout, plus finished-episode returns
    def step(c, k):
        s, o, h, first, acc = c
        ka, ke = jax.random.split(k)
        h, out = policy(th, h, o, first)
        a = jax.random.categorical(ka, out["pi"])
        s, o2, r, done = ENV_STEP(ke, s, a)
        acc = acc + r
        entropy = -(jax.nn.softmax(out["pi"]) * jax.nn.log_softmax(out["pi"])).sum(-1).mean()
        log = jnp.stack([jnp.where(done, acc, 0.0).sum(), done.sum(), entropy / m.ROLL])  # entropy: rollout mean
        return (s, o2, h, 1.0 * done, jnp.where(done, 0.0, acc)), (
            dict(s=o, first=first, a=a, r=r, done=1.0 * done, mu=out["pi"]),
            log,
        )
    new, (d, log) = jax.lax.scan(step, carry, jax.random.split(key, m.ROLL), unroll=True)
    d.update(s=jnp.concatenate([d["s"], new[1][None]]), first=jnp.concatenate([d["first"], new[3][None]]), h0=carry[2])
    return new, jax.tree.map(m.stop, d), log.sum(0)

def across(one, variants, init, key, iters, seeds):  # vmap over variants and seeds, scan over updates
    st = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), seeds))  # shared by all variants
    st = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(jax.tree.leaves(variants)[0]),) + x.shape), st)
    def it(st, k):
        ks = jax.random.split(k, seeds)
        return jax.vmap(lambda v, st: jax.vmap(lambda s, kk: one(v, s, kk))(st, ks))(variants, st)
    return jax.lax.scan(it, st, jax.random.split(jax.random.fold_in(key, 1), iters))[1]

def speed(steps, seconds, compile_s, agents):  # one log line: env steps per second, compile time, peak GPU memory
    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
    print(f"speed: {steps / seconds:,.0f} env steps/s, {steps / seconds / agents:,.0f} per agent ({agents} agents); "
          f"compile {compile_s:.0f}s; peak memory {peak:.1f} GiB", flush=True)

# ---------------------------------------------------------------- frozen learned rule
def run_rule(etas, key, iters, seeds, symbolic_rule=False):  # symbolic_rule: scripts/symbolic.py's formula
    def policy(th, h, s, first):
        return m.agent_step(th, h, s, first, "raw")
    def init(k):
        ka, ke = jax.random.split(k)
        th = m.init_agent(ka)
        zeros = jax.tree.map(jnp.zeros_like, th)
        return th, (zeros, zeros, jnp.zeros(())), init_carry(ke)
    def one(eta, st, k):
        th, opt, carry = st
        carry, d, log = rollout(policy, th, carry, k)
        th, opt = symbolic.step(th, opt, d, "raw", eta) if symbolic_rule else m.inner_step(th, opt, eta, d, "raw")[:2]
        return (th, opt, carry), log
    return across(one, etas, init, key, iters, seeds)

def load_rule(path):  # sets mini.RULE_LOGP, RULE_LEGAL, ANCHOR as the rule was trained (one run's rules must agree)
    z = np.load(path)
    m.RULE_LOGP = z["['emb']['w']"].shape[0] == 3 * m.USE_Q + 3 + m.D + 2
    m.RULE_LEGAL = bool(z["flag RULE_LEGAL"]) if "flag RULE_LEGAL" in z.files else False
    resid = "flag RULE_RESID" in z.files and bool(z["flag RULE_RESID"])  # RULE_RESID rules are ANCHOR 1
    m.ANCHOR = float(z["flag ANCHOR"]) if "flag ANCHOR" in z.files else float(resid)
    eta = m.init_rule(jax.random.PRNGKey(0))
    return jax.tree.unflatten(
        jax.tree.structure(eta),
        [jnp.asarray(z[jax.tree_util.keystr(p)]) for p, _ in jax.tree_util.tree_leaves_with_path(eta)],
    )

def load_rules(paths):  # rules that run together share mini's flags (the last load sets them): refuse a mix
    rules, flags = [], set()
    for path in paths:
        rules.append(load_rule(path))
        flags.add((m.RULE_LOGP, m.RULE_LEGAL, m.ANCHOR))
    if len(flags) > 1:
        raise ValueError(f"rules with different (RULE_LOGP, RULE_LEGAL, ANCHOR) {flags}: run them apart")
    return rules

# ---------------------------------------------------------------- PPO and A2C (purejaxrl's recurrent PPO core)
TX = optax.chain(optax.clip_by_global_norm(MAX_NORM), optax.scale_by_adam(eps=1e-5))

def base_init(k):  # mini's torso and policy head with the same weights as the rule's agent, plus a value head
    th = m.init_agent(k)
    if m.AGENT_CORE == "impala":
        return dict(impala=th["impala"], pi=th["pi"], v=m.dense(jax.random.fold_in(k, 7), m.CNN_FC, 1))
    return dict(l1=th["l1"], l2=th["l2"], gru=th["gru"], pi=th["pi"], v=m.dense(jax.random.fold_in(k, 7), m.HID, 1))

def base_policy(th, h, s, first):
    if m.AGENT_CORE == "impala":  # s: the grid times 255, then the previous action
        x = m.cnn(th["impala"], s)
        h = m.gru(th["impala"]["gru"], h * (1.0 - first)[..., None], x)
        out = m.after_gru(th["impala"], x, h)
    else:
        x = jax.nn.relu(m.apply(th["l2"], jax.nn.relu(m.apply(th["l1"], s))))
        out = h = m.gru(th["gru"], h * (1.0 - first)[..., None], x)
    return h, dict(pi=m.apply(th["pi"], out), v=m.apply(th["v"], out)[..., 0])

def base_unroll(th, d):
    return jax.lax.scan(lambda h, x: base_policy(th, h, *x), d["h0"], (d["s"], d["first"]), unroll=True)[1]

def gae(r, done, v):  # v has T+1 entries; done_t means the episode ended at step t
    disc = m.GAMMA * (1.0 - done)
    def back(g, x):
        g = x[0] + x[1] * m.LAM * g
        return g, g
    adv = jax.lax.scan(back, jnp.zeros_like(v[0]), (r + disc * v[1:] - v[:-1], disc), reverse=True)[1]
    return adv, adv + v[:-1]

def ppo_loss(th, b):
    out = jax.tree.map(lambda x: x[:-1], base_unroll(th, b))
    ratio = jnp.exp(m.pick(m.lsm(out["pi"]), b["a"])[..., 0] - b["logp"])
    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
    v_clip = b["v"] + jnp.clip(out["v"] - b["v"], -CLIP, CLIP)
    vl = 0.5 * jnp.maximum((out["v"] - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2).mean()
    return pg + VF * vl - ENT * m.entropy(out["pi"])

def run_ppo(lrs, key, iters, seeds, epochs, n_mb):
    def init(k):
        ka, ke = jax.random.split(k)
        th = base_init(ka)
        return th, TX.init(th), init_carry(ke)
    def one(lr, st, k):
        th, opt, carry = st
        kr, ku = jax.random.split(k)
        carry, d, log = rollout(base_policy, th, carry, kr)
        old = base_unroll(th, d)
        adv, ret = gae(d["r"], d["done"], old["v"])
        seq = m.stop(
            dict(
                s=d["s"],
                first=d["first"],
                a=d["a"],
                logp=m.pick(m.lsm(old["pi"][:-1]), d["a"])[..., 0],
                v=old["v"][:-1],
                adv=adv,
                ret=ret,
            )
        )
        def minibatch(c, idx):  # whole sequences of a subset of envs, from their stored memory
            th, opt = c
            b = jax.tree.map(lambda x: x[:, idx], seq) | dict(h0=d["h0"][idx])
            u, opt = TX.update(jax.grad(ppo_loss)(th, b), opt)
            return (jax.tree.map(lambda p, u: p - lr * u, th, u), opt), None
        def epoch(c, k):
            return (jax.lax.scan(minibatch, c, jax.random.permutation(k, m.B).reshape(n_mb, -1))[0], None)
        (th, opt), _ = jax.lax.scan(epoch, (th, opt), jax.random.split(ku, epochs))
        return (th, opt, carry), log
    return across(one, lrs, init, key, iters, seeds)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("env", choices=("catch", "reach", "breakout", *mn.GAMES))
    p.add_argument("arm", choices=("rule_q", "rule_noq", "symbolic", "ppo", "a2c"))
    p.add_argument("--steps", type=float, default=0, help="env steps per agent (default 1e6, breakout 5e6)")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--envs", type=int, default=m.B, help="parallel envs per agent, all arms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--actions", type=int, default=0, help="breakout: agent actions, the ones past 3 are no-ops")
    p.add_argument("--net", choices=("mlp", "impala"), default="mlp", help="ppo, a2c: agent torso (see above)")
    p.add_argument("--lrs", type=float, nargs="+", default=LRS, help="ppo, a2c: learning rates")
    a = p.parse_args()
    m.OBS, ENV_RESET, ENV_STEP = make_env(a.env)
    if a.actions:  # as rule_paper.py --actions N --pad noop: the same game, more actions
        m.A, breakout_step = a.actions, ENV_STEP
        def ENV_STEP(k, s, act):
            return breakout_step(k, s, jnp.where(act < 3, act, 0))
    if a.net == "impala":
        use_impala()
    m.B = a.envs
    per_iter = m.B * m.ROLL
    iters = int(a.steps or (5e6 if a.env == "breakout" else 1e6)) // per_iter
    key = jax.random.PRNGKey(2000 + a.seed)  # fresh agents and environments, unseen in meta-training
    if a.arm == "symbolic":
        names, variants = ["symbolic"], jnp.asarray([symbolic.COEF])
        def fn(k):
            return run_rule(variants, k, iters, a.seeds, symbolic_rule=True)
    elif a.arm.startswith("rule"):
        m.USE_Q = a.arm == "rule_q"
        run = "q" if m.USE_Q else "noq_life200"
        names, etas = [a.arm], [load_rule(RULES / run / "rule.npz")]
        if m.USE_Q:
            meta_seed = json.loads((RULES / run / "summary.json").read_text())["seed"]
            names.append("rule_untrained")
            etas.append(m.init_rule(jax.random.split(jax.random.PRNGKey(meta_seed), 3)[0]))
        variants = jax.tree.map(lambda *x: jnp.stack(x), *etas)
        def fn(k):
            return run_rule(variants, k, iters, a.seeds)
    else:
        epochs, n_mb = (4, 4) if a.arm == "ppo" else (1, 1)
        names = [f"{a.arm}{' impala' * (a.net == 'impala')} lr={lr:g}" for lr in a.lrs]
        def fn(k):
            return run_ppo(jnp.asarray(a.lrs), k, iters, a.seeds, epochs, n_mb)
    start = time.monotonic()
    compiled = jax.jit(fn).lower(key).compile()
    compile_s = time.monotonic() - start
    log = np.asarray(jax.block_until_ready(compiled(key)))  # [iters, variants, seeds, (return sum, episodes)]
    run_s = time.monotonic() - start - compile_s
    out = a.out / a.env
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{a.arm}.npz",
        log=log,
        names=np.array(names),
        steps_per_iter=per_iter,
        envs=a.envs,
        compile_s=compile_s,
        run_s=run_s,
        device=str(jax.devices()[0]),
    )
    tail = log[-max(1, iters // 10) :].sum(0)
    for i, n in enumerate(names):
        r = tail[i, :, 0] / np.maximum(tail[i, :, 1], 1)
        print(
            f"{a.env:8s} {n:22s} last 10%: {r.mean():+7.3f} +- {r.std() / np.sqrt(len(r)):.3f}"
            f"  ({iters * per_iter / 1e6:.1f}M steps, compile {compile_s:.0f}s, run {run_s:.0f}s)",
            flush=True,
        )
