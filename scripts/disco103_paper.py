"""DeepMind's Disco103 rule on MinAtar Breakout in its own regime (paper + discocraft), agent size kept ours.

usage: python scripts/disco103_paper.py OUT [--steps 5e6] [--seeds 8] [--disco-rl PATH]
Paper (Methods): batches of 96 trajectories x 29 steps, 90% replay, Adam lr 3e-4 (held-out evals tune lr over
{1e-4, 3e-4, 5e-4}), discount 0.997. discocraft (train_paper_pixels.py), where the paper is silent: 8 envs x
29-step rollouts per update, on-policy part of the batch drawn from those 8 fresh trajectories, 400k-transition
FIFO buffer, updates start once the buffer holds one batch. Agent: 2x256 MLP torso, Disco103 heads, action model
LSTM 128 / head 128, no frame stack. Optimizer: Disco103 default (Adam with a stop-grad denominator, update
clip 1, no weight decay). All learning rates and seeds run in one jit. Writes OUT/breakout/disco103_paper.npz.
"""

import argparse
import sys
import time
from pathlib import Path

import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dm_env import specs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import breakout as bk

ENVS, T, BATCH, REPLAY, BUFFER = 8, 29, 96, 0.9, 400_000
LRS = (1e-4, 3e-4, 5e-4)
CAP = BUFFER // T  # trajectories
LAST, MID = int(dm_env.StepType.LAST), int(dm_env.StepType.MID)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--steps", type=float, default=5e6, help="env steps per agent")
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--disco-rl", type=Path, default=ROOT.parent / "discocraft" / "external" / "disco_rl")
a = p.parse_args()
sys.path.insert(0, str(a.disco_rl))
from disco_rl import agent as agent_lib  # noqa: E402
from disco_rl import optimizers, types  # noqa: E402

s = agent_lib.get_settings_disco()  # discount 0.997 for the agent and the rule's value function
s.net_settings.net_args = dict(
    dense=(256, 256),
    model_arch_name="lstm",
    head_w_init_std=1e-2,
    model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128),
)
agent = agent_lib.Agent(
    single_observation_spec={"observation": specs.Array((bk.OBS,), np.float32)},
    single_action_spec=specs.BoundedArray((), np.int32, 0, 2),
    agent_settings=s,
    batch_axis_name=None,
)
# Disco103's optimizer with the learning rate held in the optimizer state, so learning rates can be vmapped
agent._optimizer = optax.chain(
    optimizers.scale_by_adam_sg_denom(),
    optax.clip(max_delta=s.max_abs_update),
    optax.inject_hyperparams(optax.scale)(step_size=-s.learning_rate),
)
with np.load(a.disco_rl / "disco_rl" / "update_rules" / "weights" / "disco_103.npz") as f:
    flat = dict(f)
eta = {k[:-2]: {"w": flat[k], "b": flat[k[:-2] + "/b"]} for k in flat if k.endswith("/w")}  # as discocraft loads it
chex.assert_trees_all_equal_shapes_and_dtypes(agent.update_rule.init_params(jax.random.PRNGKey(0))[0], eta)

def timestep(st, reward, done):
    return types.EnvironmentTimestep(
        observation={"observation": jax.vmap(bk.obs)(st)}, step_type=jnp.where(done, LAST, MID), reward=reward
    )

def env_step(k, st, action):  # auto-reset: a LAST timestep carries the terminal reward and the new episode's first obs
    st, r, done = jax.vmap(bk.step)(st, action)
    fresh = jax.vmap(bk.reset)(jax.random.split(k, len(action)))
    st = jax.tree.map(lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), fresh, st)
    return st, timestep(st, r, done)

def collect(learner, actor, st, ts, acc, k):  # one 29-step rollout in 8 envs, as discocraft's unroll; [T, B]
    def step(c, kk):
        actor, st, ts, acc = c
        ka, ke = jax.random.split(kk)
        at, actor = agent.actor_step(learner.params, ka, ts, actor)
        st, ts = env_step(ke, st, at.actions)
        acc = acc + ts.reward
        last = ts.step_type == LAST
        log = jnp.stack([jnp.where(last, acc, 0.0).sum(), last.sum()])
        traj = dict(
            obs=at.observations["observation"].astype(jnp.uint8),
            actions=at.actions,
            rewards=at.rewards,
            discounts=at.discounts.astype(jnp.float32),
            logits=at.logits,
        )
        return (actor, st, ts, jnp.where(last, 0.0, acc)), (traj, log)
    (actor, st, ts, acc), (traj, log) = jax.lax.scan(step, (actor, st, ts, acc), jax.random.split(k, T))
    return actor, st, ts, acc, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj), log.sum(0)

def add(buf, traj):  # FIFO ring buffer of whole trajectories
    i = (buf["idx"] + jnp.arange(ENVS)) % CAP
    data = {k: buf[k].at[i].set(v) for k, v in traj.items()}
    return data | dict(idx=(buf["idx"] + ENVS) % CAP, size=jnp.minimum(CAP, buf["size"] + ENVS))

def rollout(traj):  # [B, T] trajectories -> time-major ActorRollout
    tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
    return types.ActorRollout(
        observations={"observation": tm["obs"].astype(jnp.float32)},
        actions=tm["actions"],
        rewards=tm["rewards"],
        discounts=tm["discounts"],
        agent_outs={"logits": tm["logits"]},
        states={},
        logits=tm["logits"],
    )

def init(k):
    k1, k2, k3 = jax.random.split(k, 3)
    learner, actor = agent.initial_learner_state(k1), agent.initial_actor_state(k2)
    st = jax.vmap(bk.reset)(jax.random.split(k3, ENVS))
    def zeros(shape, dt):
        return jnp.zeros((CAP,) + shape, dt)
    buf = dict(
        obs=zeros((T, bk.OBS), jnp.uint8),
        actions=zeros((T,), jnp.int32),
        rewards=zeros((T,), jnp.float32),
        discounts=zeros((T,), jnp.float32),
        logits=zeros((T, 3), jnp.float32),
        idx=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )
    return learner, actor, st, timestep(st, jnp.zeros(ENVS), jnp.zeros(ENVS, bool)), jnp.zeros(ENVS), buf

def iteration(state, k, learn=True):  # collect 8 x 29 steps, store them, one update on a 96 x 29 batch
    learner, actor, st, ts, acc, buf = state
    kc, kr, kf, km, ku = jax.random.split(k, 5)
    actor, st, ts, acc, traj, log = collect(learner, actor, st, ts, acc, kc)
    buf = add(buf, traj)
    if learn:
        replay = jax.tree.map(lambda x: x[jax.random.randint(kr, (BATCH,), 0, buf["size"])], {k: buf[k] for k in traj})
        fresh = jax.tree.map(lambda x: x[jax.random.randint(kf, (BATCH,), 0, ENVS)], traj)
        use_replay = jax.random.bernoulli(km, REPLAY, (BATCH,))
        batch = jax.tree.map(
            lambda r, f: jnp.where(use_replay.reshape((-1,) + (1,) * (r.ndim - 1)), r, f), replay, fresh
        )
        learner, _, _ = agent.learner_step(ku, rollout(batch), learner, actor, eta, False)
    return (learner, actor, st, ts, acc, buf), log

def with_lr(learner, lr):
    adam, clip, scale = learner.opt_state
    return learner.replace(opt_state=(adam, clip, scale._replace(hyperparams={"step_size": -lr})))

per_iter = ENVS * T
iters = int(a.steps) // per_iter
warmup = -(-BATCH // ENVS)  # iterations until the buffer holds one batch

def run(key):
    state = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.seeds))  # shared by all learning rates
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(LRS),) + x.shape), state)
    state = (jax.vmap(jax.vmap(with_lr, (0, None)))(state[0], jnp.asarray(LRS)),) + state[1:]
    def scan(state, n, learn, k):
        def body(st, kk):
            ks = jax.random.split(kk, a.seeds)
            return jax.vmap(lambda st: jax.vmap(lambda s1, k1: iteration(s1, k1, learn))(st, ks))(st)
        return jax.lax.scan(body, state, jax.random.split(k, n))
    state, log0 = scan(state, warmup, False, jax.random.fold_in(key, 1))
    _, log1 = scan(state, iters - warmup, True, jax.random.fold_in(key, 2))
    return jnp.concatenate([log0, log1])  # [iters, lrs, seeds, 2]

key = jax.random.PRNGKey(2000 + a.seed)
start = time.monotonic()
compiled = jax.jit(run).lower(key).compile()
compile_s = time.monotonic() - start
log = np.asarray(jax.block_until_ready(compiled(key)))
run_s = time.monotonic() - start - compile_s
out = a.out / "breakout"
out.mkdir(parents=True, exist_ok=True)
names = [f"disco103_paper lr={lr:g}" for lr in LRS]
np.savez(
    out / "disco103_paper.npz",
    log=log,
    names=np.array(names),
    steps_per_iter=per_iter,
    envs=ENVS,
    compile_s=compile_s,
    run_s=run_s,
    device=str(jax.devices()[0]),
)
tail = log[-max(1, iters // 10) :].sum(0)
for i, n in enumerate(names):
    r = tail[i, :, 0] / np.maximum(tail[i, :, 1], 1)
    print(
        f"breakout {n:26s} last 10%: {r.mean():+7.3f} +- {r.std() / np.sqrt(len(r)):.3f}"
        f"  ({iters * per_iter / 1e6:.1f}M steps, compile {compile_s:.0f}s, run {run_s:.0f}s)",
        flush=True,
    )
