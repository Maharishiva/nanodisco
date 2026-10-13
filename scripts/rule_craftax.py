"""Frozen learned rules train fresh agents on Craftax-Classic (pixels), in Disco103's paper regime as rule_paper.py.

usage: python scripts/rule_craftax.py OUT --rules A.npz B.npz ... [--labels ...] [--lrs 1e-4 3e-4 5e-4] [--seeds 4]
       [--steps 1e6] [--retrace] [--name NAME] [--envs 8 --roll 29 --batch 96 --replay 0.9 --utd 1]
Regime of rule_paper.py: per update 8 envs x 29 new steps go into a FIFO buffer, and the agent learns on a batch of 96
trajectories x 29 steps, 90% of them replayed and the rest drawn from the 8 new ones; learning starts once the buffer
holds one batch; the agent's own Adam (update clip 1) times the lr; the rule's discount 0.99 and lambda 0.95.
Replay ratio (trained steps per env step) = batch x utd / envs: 12 here. Large regime, e.g. --envs 64 --roll 32
--batch 128 (replay ratio 2), or --batch 256 --remat --cnn-steps 8 (replay ratio 4; without them it needs 22 GB more).
Env: scripts/craftax_pixels.py. Agent: mini's AGENT_CORE impala, the backbone of our Craftax PPO (IMPALA CNN 64-128-128,
fc 512, GRU 256 that also reads the previous action, a 512 layer), with mini's pi, y, z and q heads.
The buffer holds --buffer transitions per agent, as uint8 frames (12 KB each). All agents collect together (vmap); the
updates run --group agents at a time (lax.map), so memory holds --group backward passes, not one per agent.
Training runs in --chunks jit calls, and OUT/craftax/NAME.npz is rewritten after each. At the end every agent plays
--eval new episodes with its final policy (sampled actions, one episode per env, as our PPO's grade).
npz: log [iters, variants, seeds, 25]: sum of the returns of finished episodes, finished episodes, mean policy entropy,
then how often each of the 22 achievements was unlocked in them; eval_return and eval_length [variants, seeds, eval],
eval_achievements [variants, seeds, eval, 22].
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # patches mini to read stored observations
import craftax_pixels as cx
import mini as m

LRS = (1e-4, 3e-4, 5e-4)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--rules", type=Path, nargs="+", required=True)
p.add_argument("--labels", nargs="+", help="one per rule (default: parent dir/file stem)")
p.add_argument("--lrs", type=float, nargs="+", default=LRS)
p.add_argument("--seeds", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=float, default=1e6, help="env steps per agent")
p.add_argument("--retrace", action="store_true", help="Retrace Q targets, for rules meta-trained with mini v3")
p.add_argument("--envs", type=int, default=8, help="envs per agent: new trajectories per update")
p.add_argument("--roll", type=int, default=29, help="steps per rollout (trajectory length)")
p.add_argument("--unroll", type=int, default=1, help="time steps per loop iteration in mini's scans (0: all)")
p.add_argument("--cnn-steps", type=int, default=0, help="time steps per CNN pass in updates (mini.CNN_STEPS; 0: all)")
p.add_argument("--remat", action="store_true", help="recompute the CNN blocks in the backward pass (mini.CNN_REMAT): "
               "about 1/10 of the memory per trained frame, one more forward pass (for --batch 256 and more)")
p.add_argument("--batch", type=int, default=96, help="trajectories per update")
p.add_argument("--replay", type=float, default=0.9, help="replayed share of the batch")
p.add_argument("--utd", type=int, default=1, help="updates per rollout (replay ratio), each on its own batch")
p.add_argument("--buffer", type=int, default=50_000, help="transitions per agent")
p.add_argument("--group", type=int, default=1, help="agents per update pass")
p.add_argument("--micro", type=int, default=1, help="gradient parts per batch: less memory, one more forward pass")
p.add_argument("--chunks", type=int, default=10, help="jit calls for the training; the npz is saved after each")
p.add_argument("--eval", type=int, default=256, help="final-policy episodes per agent (0: none)")
p.add_argument("--gamma", type=float, default=m.GAMMA, help="discount of the Q targets (rule inputs v, adv)")
p.add_argument("--name", default="rule_q_paper")
p.add_argument("--cache", type=Path, help="persistent compile cache")
a = p.parse_args()
ENVS, T, m.UNROLL, m.CNN_REMAT, m.CNN_STEPS = a.envs, a.roll, a.unroll or True, a.remat, a.cnn_steps
if a.cache:
    jax.config.update("jax_compilation_cache_dir", str(a.cache))
m.AGENT_CORE = "impala"
m.use_env("craftax", cx)  # OBS and A (17)
m.B, m.ROLL, m.BATCH, m.REPLAY, m.BUFFER, m.UTD = ENVS, T, a.batch, a.replay, a.buffer, a.utd
m.USE_Q, m.RETRACE, m.AGENT_LR = True, a.retrace, 1.0  # lr 1: inner_step returns the raw Adam step, scaled per agent
m.GAMMA = a.gamma
V, S = len(a.rules) * len(a.lrs), a.seeds  # variants (rule, lr), seeds; agent i is variant i // S, seed i % S
N = V * S
rules = [eta for eta in bench.load_rules(a.rules) for _ in a.lrs]
etas = jax.tree.map(lambda *x: jnp.repeat(jnp.stack(x), S, 0), *rules)
lrs = jnp.repeat(jnp.asarray([lr for _ in a.rules for lr in a.lrs]), S)

def init(key):  # one agent: params, Adam state, carry (env state, memory, first flags, running returns), buffer
    key_agent, key_env = jax.random.split(key)
    th = m.init_agent(key_agent)
    zeros = jax.tree.map(jnp.zeros_like, th)
    carry = (cx.reset(key_env, 0, ENVS), jnp.zeros((ENVS, m.memory_size())), jnp.ones(ENVS), jnp.zeros(ENVS))
    buf = {k: v for k, v in m.init_buffer("craftax").items() if k != "valid"}  # all actions are legal
    return th, (zeros, zeros, jnp.zeros(())), carry, buf

def rollout(th, carry, key):  # T steps in ENVS envs, recorded as mini.collect does; logs finished episodes
    def step(carry, key):
        state, memory, first, ret = carry
        key_action, key_env = jax.random.split(key)
        s = cx.obs(state)
        memory, out = m.agent_step(th, memory, s, first, "craftax")
        action = jax.random.categorical(key_action, out["pi"])
        state, r, done = cx.step(key_env, state, action)
        ret, done = ret + r, 1.0 * done
        log = jnp.concatenate([jnp.stack([(done * ret).sum(), done.sum(), m.entropy(out["pi"]) / T]),
                               state["ended"].sum(0)])
        record = dict(s=s, first=first, a=action, r=r, done=done, mu=out["pi"])
        return (state, memory, done, (1 - done) * ret), (record, log)
    new, (d, log) = jax.lax.scan(step, carry, jax.random.split(key, T))
    d |= dict(s=jnp.concatenate([d["s"], cx.obs(new[0])[None]]), first=jnp.concatenate([d["first"], new[2][None]]),
              h0=carry[1])
    return new, jax.tree.map(m.stop, d), log.sum(0)

def update(th, opt, eta, lr, batch):  # one update by the frozen rule: its raw Adam step times this agent's lr
    if a.micro == 1:
        new, opt = m.frozen_step(th, opt, eta, batch, "craftax")
    else:  # as inner_step, but the gradient is the mean over a.micro parts of the batch (the targets see all of it)
        outputs = m.unroll(th, batch, "craftax")
        values = m.q_stats(outputs, batch)
        targets = m.anchor(m.rule(eta, *m.features(outputs, batch, values), batch), outputs) + (values["g"],)
        def split(x, axis=1):  # trajectory axis -> leading parts axis
            return jnp.moveaxis(x.reshape(x.shape[:axis] + (a.micro, -1) + x.shape[axis + 1 :]), axis, 0)
        parts = jax.tree.map(split, targets), {k: split(v, 0 if k == "h0" else 1) for k, v in batch.items()}
        grads = jax.lax.map(lambda x: jax.grad(m.inner_loss)(th, *x, "craftax"), parts)
        new, opt = m.adam(th, opt, jax.tree.map(lambda g: g.mean(0), grads))
    return jax.tree.map(lambda p, q: p + lr * (q - p), th, new), opt

@functools.partial(jax.jit, static_argnums=(4, 5), donate_argnums=0)
def advance(state, etas, lrs, key, n, learn):  # n iterations: every agent collects and stores, then learns
    def iteration(state, key):
        th, opt, carry, buf = state
        key_collect, key_sample = jax.random.split(key)
        carry, d, log = jax.vmap(rollout)(th, carry, jax.random.split(key_collect, N))
        buf = jax.vmap(m.store)(buf, d)
        def learn_once(th_opt, key):  # every agent learns on a new batch
            batch = jax.vmap(m.sample)(buf, d, jax.random.split(key, N))
            return jax.lax.map(lambda x: update(*x), (*th_opt, etas, lrs, batch), batch_size=a.group), None
        if learn:
            th, opt = jax.lax.scan(learn_once, (th, opt), m.update_keys(key_sample))[0]
        return (th, opt, carry, buf), log
    return jax.lax.scan(iteration, state, jax.random.split(key, n))

@jax.jit
@jax.vmap
def evaluate(th, key):  # a.eval new episodes of one agent's final policy, sampled actions, until all have ended
    def body(c):
        state, memory, first, over, ret, length, ach, key = c
        key, key_action, key_env = jax.random.split(key, 3)
        memory, out = m.agent_step(th, memory, cx.obs(state), first, "craftax")
        state, r, done = cx.step(key_env, state, jax.random.categorical(key_action, out["pi"]))
        live = ~over
        ach = jnp.where((live & done)[:, None], state["ended"], ach)
        return state, memory, 0 * first, over | done, ret + live * r, length + live, ach, key
    key_reset, key = jax.random.split(key)
    n = a.eval
    c = (cx.reset(key_reset, 0, n), jnp.zeros((n, m.memory_size())), jnp.ones(n), jnp.zeros(n, bool), jnp.zeros(n),
         jnp.zeros(n, jnp.int32), jnp.zeros((n, cx.ACHIEVEMENTS), bool), key)
    return jax.lax.while_loop(lambda c: ~c[3].all(), body, c)[4:7]

per_iter = ENVS * T
iters = round(a.steps / per_iter)
warmup = -(-a.batch // ENVS)  # iterations before the buffer holds one batch
chunk = -(-(iters - warmup) // a.chunks)
labels = [f"{x} " for x in a.labels or [f"{r.parent.name}/{r.stem}" for r in a.rules]]
names = [f"{a.name} {label}lr={lr:g}" for label in labels for lr in a.lrs]
out = a.out / "craftax"
out.mkdir(parents=True, exist_ok=True)

def save(logs, **extra):
    log = np.concatenate(logs).reshape((-1, V, S) + logs[0].shape[2:])
    np.savez(out / f"{a.name}.npz", log=log, names=np.array(names), steps_per_iter=per_iter, envs=ENVS, batch=a.batch,
             replay=a.replay, utd=a.utd, buffer=a.buffer, retrace=a.retrace, gamma=a.gamma,
             device=str(jax.devices()[0]), **extra)
    return log

key = jax.random.PRNGKey(2000 + a.seed)  # fresh agents and worlds; seed j starts the same for every variant
agent_keys = jnp.tile(jax.random.split(jax.random.fold_in(key, 0), S), (V, 1))
start = time.monotonic()
state = jax.jit(jax.vmap(init))(agent_keys)
collect_only, learn = (advance.lower(state, etas, lrs, key, *x).compile() for x in ((warmup, False), (chunk, True)))
memory, compile_s, learned_s = learn.memory_analysis().temp_size_in_bytes / 2**30, time.monotonic() - start, 0.0
print(f"{N} agents, buffers of {a.buffer} transitions: {state[3]['s'].nbytes / 2**30:.1f} GiB; compile "
      f"{compile_s:.0f}s; a learning chunk needs {memory:.1f} GiB more", flush=True)
state, log = collect_only(state, etas, lrs, jax.random.fold_in(key, 1))
logs = [np.asarray(log)]
for c in range(a.chunks):
    t = time.monotonic()
    state, log = learn(state, etas, lrs, jax.random.fold_in(key, 10 + c))
    logs.append(np.asarray(log))
    learned_s += time.monotonic() - t
    log = save(logs, run_s=time.monotonic() - start)
    r = log[-chunk:].sum(0)
    ret = (r[..., 0] / np.maximum(r[..., 1], 1)).mean(-1)
    print(f"{len(log) * per_iter / 1e6:5.2f}M steps, {time.monotonic() - start:6.0f}s, return per variant "
          + " ".join(f"{x:5.2f}" for x in ret) + f", entropy {log[-chunk:, ..., 2].sum(0).mean() / chunk:.2f}",
          flush=True)
extra = {}
if a.eval:
    eval_keys = jnp.tile(jax.random.split(jax.random.fold_in(key, 2), S), (V, 1))
    ret, length, ach = (np.asarray(x).reshape((V, S) + x.shape[1:]) for x in evaluate(state[0], eval_keys))
    extra = dict(eval_return=ret, eval_length=length, eval_achievements=ach)
log = save(logs, run_s=time.monotonic() - start, **extra)
tail = log[-max(1, len(log) // 10) :].sum(0)
for i, n in enumerate(names):
    r = tail[i, :, 0] / np.maximum(tail[i, :, 1], 1)
    line = f"craftax {n:40s} training return, last 10%: {r.mean():6.2f} +- {r.std() / np.sqrt(S):.2f}"
    if a.eval:
        e = extra["eval_return"][i].mean(-1)
        line += (f"; final policy {e.mean():6.2f} +- {e.std() / np.sqrt(S):.2f}"
                 f" ({a.eval} episodes x {S} seeds, length {extra['eval_length'][i].mean():.0f})")
    print(line, flush=True)
print(f"{len(log) * per_iter / 1e6:.2f}M steps per agent, {time.monotonic() - start:.0f}s", flush=True)
bench.speed(N * per_iter * chunk * a.chunks, learned_s, compile_s, N)
