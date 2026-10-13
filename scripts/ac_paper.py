"""PPO or A2C on MinAtar Breakout in Disco103's paper regime, exactly as scripts/rule_paper.py runs our rule.

usage: python scripts/ac_paper.py OUT ppo|a2c [--steps 5e6] [--seeds 8]
Regime: 8 envs x 29-step rollouts per update, batches of 96 trajectories x 29 steps, 90% replay, on-policy part
drawn from the 8 fresh trajectories, 400k-transition FIFO buffer, updates start once the buffer holds one batch.
Learner: bench.py's PPO/A2C (same torso, GAE gamma 0.99 lambda 0.95, clip 0.2, value 0.5, entropy 0.01, grad norm
0.5, Adam eps 1e-5). No off-policy correction, as for our rule: PPO's old policy and GAE values come from the
current network at update time. PPO: 4 epochs x 4 minibatches of 24 trajectories per batch; A2C: 1 x 1.
lr sweep {1e-4, 3e-4, 5e-4, 1e-3}. Writes OUT/breakout/{ppo,a2c}_paper.npz.
--epochs/--minibatches set PPO's reuse, e.g. 1 x 4 to match our rule's samples per update.
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

ENVS, T, BATCH, REPLAY, BUFFER = 8, 29, 96, 0.9, 400_000
LRS = (1e-4, 3e-4, 5e-4, 1e-3)
CAP = BUFFER // T

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("arm", choices=("ppo", "a2c"))
p.add_argument("--steps", type=float, default=5e6, help="env steps per agent")
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=0, help="PPO epochs per batch (default 4)")
p.add_argument("--minibatches", type=int, default=0, help="PPO minibatches per epoch (default 4)")
a = p.parse_args()
m.OBS, bench.ENV_RESET, bench.ENV_STEP = bench.make_env("breakout")
m.B, m.ROLL = ENVS, T
EPOCHS, N_MB = (a.epochs or 4, a.minibatches or 4) if a.arm == "ppo" else (1, 1)

def unroll(th, d):  # bench.base_unroll without unrolling the scan (faster compile inside the epoch loops)
    return jax.lax.scan(lambda h, x: bench.base_policy(th, h, *x), d["h0"], (d["s"], d["first"]))[1]

def loss(th, b):  # bench.ppo_loss
    out = jax.tree.map(lambda x: x[:-1], unroll(th, b))
    ratio = jnp.exp(m.pick(m.lsm(out["pi"]), b["a"])[..., 0] - b["logp"])
    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - bench.CLIP, 1 + bench.CLIP) * adv).mean()
    v_clip = b["v"] + jnp.clip(out["v"] - b["v"], -bench.CLIP, bench.CLIP)
    vl = 0.5 * jnp.maximum((out["v"] - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2).mean()
    return pg + bench.VF * vl - bench.ENT * m.entropy(out["pi"])

def learn(th, opt, d, lr, k):  # bench.run_ppo's update on one time-major batch
    old = unroll(th, d)
    adv, ret = bench.gae(d["r"], d["done"], old["v"])
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
    def minibatch(c, idx):
        th, opt = c
        b = jax.tree.map(lambda x: x[:, idx], seq) | dict(h0=d["h0"][idx])
        u, opt = bench.TX.update(jax.grad(loss)(th, b), opt)
        return (jax.tree.map(lambda p, u: p - lr * u, th, u), opt), None
    def epoch(c, k):
        return (jax.lax.scan(minibatch, c, jax.random.permutation(k, BATCH).reshape(N_MB, -1))[0], None)
    return jax.lax.scan(epoch, (th, opt), jax.random.split(k, EPOCHS))[0]

def init(k):
    ka, ke = jax.random.split(k)
    th = bench.base_init(ka)
    def z(shape, dt=jnp.float32):
        return jnp.zeros((CAP,) + shape, dt)
    buf = dict(
        s=z((T + 1, m.OBS), jnp.uint8),
        first=z((T + 1,)),
        a=z((T,), jnp.int32),
        r=z((T,)),
        done=z((T,)),
        mu=z((T, m.A)),
        h0=z((m.HID,)),
        idx=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )
    return th, bench.TX.init(th), bench.init_carry(ke), buf

def iteration(lr, state, k, learning=True):  # as rule_paper.iteration
    th, opt, carry, buf = state
    kc, kr, kf, km, ku = jax.random.split(k, 5)
    carry, d, log = bench.rollout(bench.base_policy, th, carry, kc)
    traj = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in d.items()}
    traj["s"] = traj["s"].astype(jnp.uint8)
    i = (buf["idx"] + jnp.arange(ENVS)) % CAP
    buf = {k: buf[k].at[i].set(v) for k, v in traj.items()} | dict(
        idx=(buf["idx"] + ENVS) % CAP, size=jnp.minimum(CAP, buf["size"] + ENVS)
    )
    if learning:
        replay = jax.tree.map(lambda x: x[jax.random.randint(kr, (BATCH,), 0, buf["size"])], {k: buf[k] for k in traj})
        fresh = jax.tree.map(lambda x: x[jax.random.randint(kf, (BATCH,), 0, ENVS)], traj)
        use = jax.random.bernoulli(km, REPLAY, (BATCH,))
        batch = jax.tree.map(lambda r, f: jnp.where(use.reshape((-1,) + (1,) * (r.ndim - 1)), r, f), replay, fresh)
        batch = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in batch.items()}
        batch["s"] = batch["s"].astype(jnp.float32)
        th, opt = learn(th, opt, batch, lr, ku)
    return (th, opt, carry, buf), log

per_iter = ENVS * T
iters = int(a.steps) // per_iter
warmup = -(-BATCH // ENVS)

def run(key):
    state = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.seeds))  # shared by all learning rates
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(LRS),) + x.shape), state)
    lrs = jnp.asarray(LRS)
    def scan(state, n, learning, k):
        def body(st, kk):
            ks = jax.random.split(kk, a.seeds)
            return jax.vmap(lambda lr, st: jax.vmap(lambda s1, k1: iteration(lr, s1, k1, learning))(st, ks))(lrs, st)
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
names = [f"{a.arm}_paper {EPOCHS}x{N_MB} lr={lr:g}" for lr in LRS]
np.savez(
    out / f"{a.arm}_paper.npz",
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
