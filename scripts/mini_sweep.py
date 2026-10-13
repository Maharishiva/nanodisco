"""Meta-train mini.py rules at several scales and evaluate every checkpoint.

usage: python scripts/mini_sweep.py OUT STEPS:AGENTS[:INNER] [...]
Three meta seeds per config, ten checkpoints, four fresh-agent eval seeds.
"""

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mini as m

SEEDS, CKPTS = 3, 10
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
eval_keys = jax.random.split(jax.random.PRNGKey(10000), 4)

def flat(tree, n):
    return jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), tree)

for spec in sys.argv[2:]:
    steps, agents, *inner = map(int, spec.split(":"))
    m.INNER = inner[0] if inner else 2  # agent updates per meta-gradient (the meta horizon)
    m.LIFE = 64 // m.INNER  # keep 64 agent updates per lifetime
    name = f"steps{steps}_agents{agents}" + (f"_inner{m.INNER}" if inner else "")
    start = time.monotonic()
    run = jax.jit(jax.vmap(lambda k: m.train(k, steps, agents, CKPTS)))
    eta0, etas, losses = jax.block_until_ready(run(jax.random.split(jax.random.PRNGKey(0), SEEDS)))
    train_seconds = time.monotonic() - start
    record = dict(
        steps=steps,
        agents=agents,
        inner=m.INNER,
        seeds=SEEDS,
        train_seconds=train_seconds,
        devices=[str(d) for d in jax.devices()],
        checkpoints=[steps // CKPTS * (i + 1) for i in range(CKPTS)],
    )
    arrays = {}
    for env in ("catch", "reach"):
        def ev(n):
            return jax.jit(jax.vmap(jax.vmap(lambda e, k: m.evaluate(e, k, env, n), (None, 0)), (0, None)))
        ckpt = np.asarray(ev(60)(flat(etas, SEEDS * CKPTS), eval_keys)).reshape(SEEDS, CKPTS, 4, 60)
        final = np.asarray(ev(300)(jax.tree.map(lambda x: x[:, -1], etas), eval_keys))
        untrained = np.asarray(ev(60)(eta0, eval_keys))
        arrays.update({f"{env}_ckpt60": ckpt, f"{env}_final300": final, f"{env}_untrained60": untrained})
        record[env] = dict(
            ckpt_at60=ckpt[..., -1].mean((0, 2)).tolist(),
            final_at60=ckpt[:, -1, :, -1].mean(1).tolist(),
            final_at300=final[..., -1].mean(1).tolist(),
            untrained_at60=float(untrained[..., -1].mean()),
        )
    np.savez(
        out / f"{name}.npz", losses=np.asarray(losses), **arrays, **{f"eta_{k}": np.asarray(v) for k, v in etas.items()}
    )
    (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        name,
        f"{train_seconds:.0f}s",
        json.dumps({e: {k: np.round(v, 3).tolist() for k, v in record[e].items()} for e in ("catch", "reach")}),
        flush=True,
    )
