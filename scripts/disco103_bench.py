"""Fresh agents on MinAtar Breakout trained by DeepMind's released Disco103 rule (disco_rl), matched to bench.py.

usage: python scripts/disco103_bench.py OUT [--steps 5e6] [--envs 128] [--seeds 8] [--disco-rl PATH]
Agent: disco_rl's MLP agent with a 2x256 torso (no recurrence) and Disco103's heads (y/z of 600, 601-bin Q,
aux policy), default action model (LSTM 128, head 128). Optimizer: Disco103 default, Adam with a stop-grad
denominator, lr 3e-4, update clip 1. Data: on-policy, --envs envs x 16 steps per update, one update per
rollout, no replay; consecutive rollouts share one timestep, so every transition is learned from once.
Discount 0.99 for the agent and for the rule's value function, as in bench.py.
Writes OUT/breakout/disco103.npz in bench.py's format.
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
from dm_env import specs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import breakout as bk

ROLL = 16
LAST, MID = int(dm_env.StepType.LAST), int(dm_env.StepType.MID)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--steps", type=float, default=5e6, help="env steps per agent")
p.add_argument("--envs", type=int, default=128)
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--disco-rl", type=Path, default=ROOT.parent / "discocraft" / "external" / "disco_rl")
a = p.parse_args()
sys.path.insert(0, str(a.disco_rl))
from disco_rl import agent as agent_lib  # noqa: E402
from disco_rl import types  # noqa: E402

s = agent_lib.get_settings_disco()
s.net_settings.net_args = dict(
    dense=(256, 256),
    model_arch_name="lstm",
    head_w_init_std=1e-2,
    model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128),
)
s.hyper_params.discount_factor = 0.99
s.update_rule.value_discount = 0.99
agent = agent_lib.Agent(
    single_observation_spec={"observation": specs.Array((bk.OBS,), np.float32)},
    single_action_spec=specs.BoundedArray((), np.int32, 0, 2),
    agent_settings=s,
    batch_axis_name=None,
)
weights = a.disco_rl / "disco_rl" / "update_rules" / "weights" / "disco_103.npz"
with np.load(weights) as f:
    flat = dict(f)
eta = {k[:-2]: {"w": flat[k], "b": flat[k[:-2] + "/b"]} for k in flat if k.endswith("/w")}  # as discocraft loads it
chex.assert_trees_all_equal_shapes_and_dtypes(agent.update_rule.init_params(jax.random.PRNGKey(0))[0], eta)

def timestep(st, reward, done):
    return types.EnvironmentTimestep(
        observation={"observation": jax.vmap(bk.obs)(st)}, step_type=jnp.where(done, LAST, MID), reward=reward
    )

def env_reset(k, n):
    st = jax.vmap(bk.reset)(jax.random.split(k, n))
    return st, timestep(st, jnp.zeros(n), jnp.zeros(n, bool))

def env_step(k, st, action):  # auto-reset: a LAST timestep carries the terminal reward and the new episode's first obs
    st, r, done = jax.vmap(bk.step)(st, action)
    fresh = jax.vmap(bk.reset)(jax.random.split(k, len(action)))
    st = jax.tree.map(lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), fresh, st)
    return st, timestep(st, r, done)

def strip(t):  # keep what the learner needs, as discocraft's strip_rollout
    return types.ActorTimestep(
        observations=t.observations,
        actions=t.actions,
        rewards=t.rewards,
        discounts=t.discounts,
        agent_outs={"logits": t.logits},
        states={},
        logits=t.logits,
    )

def act(learner, actor, st, ts, acc, k):  # one actor step and env step; logs finished-episode returns
    ka, ke = jax.random.split(k)
    at, actor = agent.actor_step(learner.params, ka, ts, actor)
    st, ts = env_step(ke, st, at.actions)
    acc = acc + ts.reward
    last = ts.step_type == LAST
    return actor, st, ts, jnp.where(last, 0.0, acc), strip(at), jnp.stack([jnp.where(last, acc, 0.0).sum(), last.sum()])

def init(k):
    k1, k2, k3, k4 = jax.random.split(k, 4)
    learner, actor = agent.initial_learner_state(k1), agent.initial_actor_state(k2)
    st, ts = env_reset(k3, a.envs)
    actor, st, ts, acc, prev, _ = act(learner, actor, st, ts, jnp.zeros(a.envs), k4)
    return learner, actor, st, ts, acc, prev

def update(state, k):  # 16 new env steps per env, then one Disco103 update on the 17-timestep rollout
    learner, actor, st, ts, acc, prev = state
    kr, ku = jax.random.split(k)
    def step(c, kk):
        actor, st, ts, acc, at, log = act(learner, *c, kk)
        return (actor, st, ts, acc), (at, log)
    (actor, st, ts, acc), (roll, log) = jax.lax.scan(step, (actor, st, ts, acc), jax.random.split(kr, ROLL))
    rollout = types.ActorRollout.from_timestep(jax.tree.map(lambda x, y: jnp.concatenate([x[None], y]), prev, roll))
    learner, _, _ = agent.learner_step(ku, rollout, learner, actor, eta, False)
    return (learner, actor, st, ts, acc, jax.tree.map(lambda x: x[-1], roll)), log.sum(0)

def run(key):
    state = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.seeds))
    def step(st, k):
        return jax.vmap(update)(st, jax.random.split(k, a.seeds))
    return jax.lax.scan(step, state, jax.random.split(jax.random.fold_in(key, 1), iters))[1]

per_iter = a.envs * ROLL
iters = int(a.steps) // per_iter
key = jax.random.PRNGKey(2000 + a.seed)
start = time.monotonic()
compiled = jax.jit(run).lower(key).compile()
compile_s = time.monotonic() - start
log = np.asarray(jax.block_until_ready(compiled(key)))[:, None]  # [iters, 1 variant, seeds, (return sum, episodes)]
run_s = time.monotonic() - start - compile_s
out = a.out / "breakout"
out.mkdir(parents=True, exist_ok=True)
np.savez(
    out / "disco103.npz",
    log=log,
    names=np.array(["disco103"]),
    steps_per_iter=per_iter,
    envs=a.envs,
    compile_s=compile_s,
    run_s=run_s,
    device=str(jax.devices()[0]),
)
tail = log[-max(1, iters // 10) :].sum(0)[0]
r = tail[:, 0] / np.maximum(tail[:, 1], 1)
print(
    f"breakout disco103               last 10%: {r.mean():+7.3f} +- {r.std() / np.sqrt(len(r)):.3f}"
    f"  ({iters * per_iter / 1e6:.1f}M steps, compile {compile_s:.0f}s, run {run_s:.0f}s)",
    flush=True,
)
