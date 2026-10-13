"""Our frozen Catch-trained rule on MinAtar Breakout in Disco103's paper regime, as scripts/disco103_paper.py.

usage: python scripts/rule_paper.py OUT [--steps 5e6] [--seeds 8]
Same regime: 8 envs x 29-step rollouts per update, batches of 96 trajectories x 29 steps, 90% replay, on-policy part
drawn from the 8 fresh trajectories, 400k-transition FIFO buffer, updates start once the buffer holds one batch,
lr sweep {1e-4, 3e-4, 5e-4}. Our side: mini.py's agent (2x256 MLP + 256 GRU), the value-input rule, its own Adam
(update clip 1) and its meta-training discount 0.99 and lambda 0.95. Replayed trajectories start from the GRU
memory stored when they were collected. Writes OUT/breakout/rule_q_paper.npz.
--symbolic: the three-term formula of scripts/symbolic.py instead of the rule; writes symbolic_paper.npz.
--rules A.npz B.npz ... --lrs ...: every rule at every lr in one run (default: the value-input rule, the lrs above);
--labels names them, --name sets the output arm.
Regime knobs: --envs, --roll, --batch, --replay, --utd, --buffer; replay ratio (trained steps per env step) = batch x
utd / envs, 12 in the paper regime. Large regime, e.g. --envs 64 --roll 32 --batch 256 (replay ratio 4).
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # patches mini to read stored observations
import mini as m
import symbolic

ENVS, T, BATCH, REPLAY, BUFFER = 8, 29, 96, 0.9, 400_000
LRS = (1e-4, 3e-4, 5e-4)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("--steps", type=float, default=5e6, help="env steps per agent")
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--symbolic", action="store_true")
p.add_argument("--rules", type=Path, nargs="+", default=[bench.RULES / "q" / "rule.npz"])
p.add_argument("--lrs", type=float, nargs="+", default=LRS)
p.add_argument("--labels", nargs="+", help="one per rule (default: parent dir/file stem)")
p.add_argument("--name")
p.add_argument("--retrace", action="store_true", help="Retrace Q targets, for rules meta-trained with mini v3")
p.add_argument("--envs", type=int, default=ENVS, help="new trajectories per update")
p.add_argument("--roll", type=int, default=T, help="steps per rollout (trajectory length)")
p.add_argument("--batch", type=int, default=BATCH, help="trajectories per update")
p.add_argument("--replay", type=float, default=REPLAY, help="replayed share of the batch")
p.add_argument("--buffer", type=int, default=BUFFER, help="replay buffer size in transitions")
p.add_argument("--utd", type=int, default=1, help="updates per rollout (replay ratio), each on its own batch")
p.add_argument("--unroll", type=int, default=1, help="time steps per loop iteration in mini's scans (0: all)")
p.add_argument("--cache", type=Path, help="persistent compile cache: later runs with the same shapes skip the compile")
p.add_argument("--agent-core", default=m.AGENT_CORE, choices=("gru", "mlp", "gdn", "impala"),
               help="mini.AGENT_CORE; impala: our Craftax PPO agent on the grid (bench.use_impala)")
p.add_argument("--rule-core", default=m.RULE_CORE, choices=("gru", "attn", "gdn"), help="mini.RULE_CORE, of the rules")
p.add_argument("--assoc", action="store_true", help="returns by associative scans (mini.ASSOC)")
p.add_argument("--game", default="breakout", help="breakout, or a MinAtar game of scripts/minatar.py (its actions)")
p.add_argument("--actions", type=int, default=3, help="agent actions; the ones past Breakout's 3 act as --pad says")
p.add_argument("--pad", default="noop", choices=("noop", "copy"), help="extra actions: no-ops, or copies (a mod 3)")
a = p.parse_args()
m.AGENT_CORE, m.RULE_CORE, m.ASSOC = a.agent_core, a.rule_core, a.assoc
if a.cache:
    jax.config.update("jax_compilation_cache_dir", str(a.cache))
m.RETRACE, m.UNROLL, m.UTD, ENVS, BATCH = a.retrace, a.unroll or True, a.utd, a.envs, a.batch
T, REPLAY, CAP = a.roll, a.replay, a.buffer // a.roll
m.OBS, bench.ENV_RESET, bench.ENV_STEP = bench.make_env(a.game)
if a.actions != 3:  # same game, more actions: does a rule meta-trained on 3-6 actions cope with Craftax's 17?
    m.A, breakout_step = a.actions, bench.ENV_STEP
    def padded_step(k, s, act):
        return breakout_step(k, s, jnp.where(act < 3, act, 0) if a.pad == "noop" else act % 3)
    bench.ENV_STEP = padded_step
if a.agent_core == "impala":
    bench.use_impala()
m.B, m.ROLL, m.USE_Q, m.AGENT_LR = (
    ENVS,
    T,
    True,
    1.0,
)  # lr 1: inner_step returns the raw Adam step, scaled per agent below
rules = [eta for eta in bench.load_rules(a.rules) for _ in a.lrs]  # one variant per (rule, lr)
etas = jax.tree.map(lambda *x: jnp.stack(x), *rules)
lrs = jnp.asarray([lr for _ in a.rules for lr in a.lrs])

def policy(th, h, s, first):
    return m.agent_step(th, h, s, first, "raw")

def init(k):
    ka, ke = jax.random.split(k)
    th = m.init_agent(ka)
    zeros = jax.tree.map(jnp.zeros_like, th)
    def z(shape, dt=jnp.float32):
        return jnp.zeros((CAP,) + shape, dt)
    buf = dict(
        s=z((T + 1, m.OBS), jnp.uint8),
        first=z((T + 1,)),
        a=z((T,), jnp.int32),
        r=z((T,)),
        done=z((T,)),
        mu=z((T, m.A)),
        h0=z((m.memory_size(),)),
        idx=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )
    return th, (zeros, zeros, jnp.zeros(())), bench.init_carry(ke), buf

def iteration(eta, lr, state, k, learn=True):  # collect 8 x 29 steps, store them, --utd updates on 96 x 29 batches
    th, opt, carry, buf = state
    kc, kr, kf, km = jax.random.split(k, 4)
    carry, d, log = bench.rollout(policy, th, carry, kc)
    traj = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in d.items()}  # batch-major
    traj["s"] = traj["s"].astype(jnp.uint8)
    i = (buf["idx"] + jnp.arange(ENVS)) % CAP
    buf = {k: buf[k].at[i].set(v) for k, v in traj.items()} | dict(
        idx=(buf["idx"] + ENVS) % CAP, size=jnp.minimum(CAP, buf["size"] + ENVS)
    )
    def update(th_opt, keys):
        (th, opt), (kr, kf, km) = th_opt, keys
        replay = jax.tree.map(lambda x: x[jax.random.randint(kr, (BATCH,), 0, buf["size"])], {k: buf[k] for k in traj})
        fresh = jax.tree.map(lambda x: x[jax.random.randint(kf, (BATCH,), 0, ENVS)], traj)
        use = jax.random.bernoulli(km, REPLAY, (BATCH,))
        batch = jax.tree.map(lambda r, f: jnp.where(use.reshape((-1,) + (1,) * (r.ndim - 1)), r, f), replay, fresh)
        batch = {k: (v if k == "h0" else jnp.swapaxes(v, 0, 1)) for k, v in batch.items()}  # time-major
        batch["s"] = batch["s"].astype(jnp.float32)
        if a.symbolic:
            new, opt = symbolic.step(th, opt, batch, "raw")
        else:
            new, opt = m.frozen_step(th, opt, eta, batch, "raw")
        return (jax.tree.map(lambda p, q: p + lr * (q - p), th, new), opt), None
    if learn:  # keys [utd, 3]: --utd 1 uses kr, kf, km as before
        th, opt = jax.lax.scan(update, (th, opt), jax.vmap(m.update_keys, out_axes=1)(jnp.stack([kr, kf, km])))[0]
    return (th, opt, carry, buf), log

per_iter = ENVS * T
iters = int(a.steps) // per_iter
warmup = -(-BATCH // ENVS)

def run(key, etas, lrs):  # rules and lrs are inputs, not constants, so the cache serves any rule
    state = jax.vmap(init)(jax.random.split(jax.random.fold_in(key, 0), a.seeds))  # shared by all variants
    state = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(lrs),) + x.shape), state)
    def scan(state, n, learn, k):
        def body(st, kk):
            ks = jax.random.split(kk, a.seeds)
            def variant(eta, lr, st):
                return jax.vmap(lambda s1, k1: iteration(eta, lr, s1, k1, learn))(st, ks)
            return jax.vmap(variant)(etas, lrs, st)
        return jax.lax.scan(body, state, jax.random.split(k, n))
    state, log0 = scan(state, warmup, False, jax.random.fold_in(key, 1))
    _, log1 = scan(state, iters - warmup, True, jax.random.fold_in(key, 2))
    return jnp.concatenate([log0, log1])  # [iters, variants, seeds, 2]

key = jax.random.PRNGKey(2000 + a.seed)
start = time.monotonic()
compiled = jax.jit(run).lower(key, etas, lrs).compile()
compile_s = time.monotonic() - start
log = np.asarray(jax.block_until_ready(compiled(key, etas, lrs)))
run_s = time.monotonic() - start - compile_s
out = a.out / "breakout"
out.mkdir(parents=True, exist_ok=True)
arm = a.name or ("symbolic_paper" if a.symbolic else "rule_q_paper")
labels = [f"{x} " for x in a.labels or [f"{r.parent.name}/{r.stem}" for r in a.rules]] if len(a.rules) > 1 else [""]
names = [f"{arm} {label}lr={lr:g}" for label in labels for lr in a.lrs]
np.savez(
    out / f"{arm}.npz",
    log=log,
    names=np.array(names),
    steps_per_iter=per_iter,
    envs=ENVS,
    utd=a.utd,
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
bench.speed(len(lrs) * a.seeds * iters * per_iter, run_s, compile_s, len(lrs) * a.seeds)
