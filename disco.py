"""nanoDisco: learned KL update targets, not a hand-written policy gradient.

Agent theta sees pixels; reverse-GRU rule eta sees transitions/predictions.
Separate learned pixel critic supplies stopped outer advantages. No oracle.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

A, D, H, WIDTH, STEPS = 3, 4, 16, 5, 4
softmax, stop = jax.nn.softmax, jax.lax.stop_gradient
tree = jax.tree.map

def pixels(s, env):
    """5x5 two-channel board: object/goal and player. No coordinates as input."""
    t, goal, player = s.T
    board = jnp.zeros((len(s), STEPS + 1, WIDTH, 2))
    rows = t if env == "catch" else jnp.ones_like(t)
    return (
        board.at[jnp.arange(len(s)), rows, goal, 0]
        .set(1)
        .at[jnp.arange(len(s)), STEPS if env == "catch" else 1, player, 1]
        .set(1)
    )

def transition(s, a, env):
    t, goal, player = jnp.moveaxis(s, -1, 0)
    player = jnp.clip(player + a - 1, 0, WIDTH - 1)
    t, goal = jnp.broadcast_to(t, player.shape), jnp.broadcast_to(goal, player.shape)
    hit, timeout = player == goal, t == STEPS - 1
    done = timeout if env == "catch" else hit | timeout
    reward = jnp.where(timeout, jnp.where(hit, 1.0, -1.0), 0.0) if env == "catch" else hit.astype(jnp.float32)
    return jnp.stack((jnp.minimum(t + 1, STEPS - 1), goal, player), -1), reward, done

def reset(key, batch, env):
    goal = jax.random.randint(key, (batch,), 0, WIDTH if env == "catch" else 2)
    goal = goal if env == "catch" else (WIDTH - 1) * goal
    return jnp.stack((jnp.zeros(batch, jnp.int32), goal, jnp.full(batch, WIDTH // 2, jnp.int32)), -1)

def init_agent(key, q=True, next_policy=True):
    keys = iter(jax.random.split(key, 7))
    def weight(shape, scale):
        return scale * jax.random.normal(next(keys), shape)
    theta = dict(
        body=weight(((STEPS + 1) * WIDTH * 2, 32), 0.25),
        bias=jnp.zeros(32),
        pi=weight((32, A), 0.02),
        y=weight((32, D), 0.1),
        z=weight((32, A * D), 0.1),
    )
    if q:
        theta["q"] = weight((32, A), 0.02)
    if next_policy:
        theta["next"] = weight((32, A * A), 0.02)
    return theta

def agent(theta, s, env):
    h = jnp.tanh(pixels(s.reshape(-1, 3), env).reshape(-1, theta["body"].shape[0]) @ theta["body"] + theta["bias"])
    out = {k: h @ v for k, v in theta.items() if k not in ("body", "bias")}
    out["z"] = out["z"].reshape(-1, A, D)
    if "next" in out:
        out["next"] = out["next"].reshape(-1, A, A)
    return tree(lambda x: x.reshape(s.shape[:-1] + x.shape[1:]), out)

def collect(theta, s, key, env, length=2):
    """On-policy short chunks with learned next-Y/Z bootstrap inputs."""
    def step(s, key):
        ka, kr = jax.random.split(key)
        mu = stop(agent(theta, s, env)["pi"])
        a = jax.random.categorical(ka, mu)
        ns, r, done = transition(s, a, env)
        carry = jnp.where(done[:, None], reset(kr, len(s), env), ns)
        return carry, dict(s=s, ns=ns, a=a, r=r, done=done, mu=mu)
    return jax.lax.scan(step, s, jax.random.split(key, length))

def kl(teacher, student):
    p = softmax(teacher)
    return (p * (jax.nn.log_softmax(teacher) - jax.nn.log_softmax(student))).sum(-1)

def take(x, a):
    return jnp.take_along_axis(x, a[..., None, None], -2).squeeze(-2)

def features(out, nxt, ema, ema_next, data):
    p, ep = softmax(out["pi"]), softmax(ema["pi"])
    chosen = jax.nn.one_hot(data["a"], A)
    # No reward-to-return formula here. Y/Z have no predefined interpretation.
    common = [data["r"][..., None], (1.0 - data["done"])[..., None]]
    for current, future in ((out, nxt), (ema, ema_next)):
        common += [softmax(current["y"]), softmax(future["y"])]
        for o in (current, future):
            z, pi = softmax(o["z"]), stop(softmax(o["pi"]))
            common += [(z * pi[..., None]).sum(-2), z.max(-2)]
        common += [take(softmax(current["z"]), data["a"])]
    base = jnp.concatenate(common, -1)
    parts = [
        jnp.broadcast_to(base[..., None, :], base.shape[:-1] + (A, base.shape[-1])),
        stop(p[..., None]),
        stop(softmax(data["mu"])[..., None]),
        stop(ep[..., None]),
        chosen[..., None],
        softmax(out["z"]),
        softmax(ema["z"]),
    ]
    if "q" in out:
        v = (ep * ema["q"]).sum(-1)
        v_next = (softmax(ema_next["pi"]) * ema_next["q"]).sum(-1)
        td = data["r"] + 0.99 * (1.0 - data["done"]) * v_next
        # Q switch also removes every explicit value/advantage feature.
        parts += [
            stop(out["q"])[..., None],
            stop(ema["q"] - v[..., None])[..., None],
            jnp.broadcast_to(stop(td - v)[..., None, None], p.shape + (1,)),
        ]
    return jnp.concatenate(parts, -1)

def init_rule(key, f):
    kx, kh, ko = jax.random.split(key, 3)
    return dict(
        wx=0.15 * jax.random.normal(kx, (f, 3 * H)),
        wh=0.15 * jax.random.normal(kh, (H, 3 * H)),
        b=jnp.zeros(3 * H),
        out=0.02 * jax.random.normal(ko, (H, 1 + 2 * D)),
    )

def rule(eta, x, done):
    """A shared per-action reverse GRU; reset at episode boundaries."""
    def cell(h, inputs):
        x, terminal = inputs
        h = h * (1.0 - terminal)[..., None, None]
        xr, xu, xc = jnp.split(x @ eta["wx"] + eta["b"], 3, -1)
        hr, hu, hc = jnp.split(h @ eta["wh"], 3, -1)
        r, u = jax.nn.sigmoid(xr + hr), jax.nn.sigmoid(xu + hu)
        h = u * h + (1 - u) * jnp.tanh(xc + r * hc)
        return h, h @ eta["out"]
    _, logits = jax.lax.scan(cell, jnp.zeros(x.shape[1:3] + (H,)), (x, done), reverse=True)
    # Direct learned targets. EMA supplies inputs and the outer KL regularizer.
    return (logits[..., 0], logits[..., 1 : 1 + D].mean(-2), logits[..., 1 + D :].mean(-2))

def update(theta, ema_theta, eta, data, env, lr):
    out, nxt = agent(theta, data["s"], env), agent(theta, data["ns"], env)
    ema, ema_next = agent(ema_theta, data["s"], env), agent(ema_theta, data["ns"], env)
    targets = rule(eta, features(out, nxt, ema, ema_next, data), data["done"])
    # Targets are OUTSIDE grad(theta): fixed for the inner gradient, live for eta.
    def loss(th):
        o = agent(th, data["s"], env)
        loss = kl(targets[0], o["pi"]) + kl(targets[1], o["y"]) + kl(targets[2], take(o["z"], data["a"]))
        if "q" in o:
            v_next = (softmax(ema_next["pi"]) * ema_next["q"]).sum(-1)
            q_target = stop(data["r"] + 0.99 * (1.0 - data["done"]) * v_next)
            qa = (o["q"] * jax.nn.one_hot(data["a"], A)).sum(-1)
            loss += 0.2 * (qa - q_target) ** 2
        if "next" in o:
            loss += (1.0 - data["done"]) * kl(stop(nxt["pi"]), take(o["next"], data["a"]))
        return loss.mean()
    grad = jax.grad(loss)(theta)
    scale = jnp.maximum(1.0, optax.tree.norm(grad))
    theta_new = tree(lambda p, g: p - lr * g / scale, theta, grad)
    ema_new = tree(lambda e, p: 0.9 * e + 0.1 * p, ema_theta, theta)
    regularizer = 0.01 * kl(stop(ema["pi"]), targets[0]).mean() + 0.001 * sum(t.mean() ** 2 for t in targets)
    return theta_new, ema_new, regularizer

def init_critic(key):
    k1, k2 = jax.random.split(key)
    return dict(
        w=0.25 * jax.random.normal(k1, ((STEPS + 1) * WIDTH * 2, 32)),
        b=jnp.zeros(32),
        v=0.01 * jax.random.normal(k2, (32,)),
        c=jnp.zeros(()),
    )

def critic(phi, s, env):
    x = pixels(s.reshape(-1, 3), env).reshape(-1, phi["w"].shape[0])
    return (jax.nn.relu(x @ phi["w"] + phi["b"]) @ phi["v"] + phi["c"]).reshape(s.shape[:-1])

def decode_value(raw):
    """Stable inverse of the reference signed-hyperbolic value transform."""
    z = 2 * (jnp.abs(raw) + 1.001) / (1 + jnp.sqrt(1 + 0.004 * (jnp.abs(raw) + 1.001)))
    return jnp.sign(raw) * (z * z - 1)

def normalize(x, stats, subtract=True):
    """Reference-style debiased EMA mean/variance, not a batch baseline."""
    m, m2, power = stats
    stats = 0.99 * m + 0.01 * x.mean(), 0.99 * m2 + 0.01 * (x * x).mean(), 0.99 * power
    mean = stats[0] / (1 - stats[2])
    var = jnp.maximum(0.0, stats[1] / (1 - stats[2]) - mean**2)
    return (x - mean if subtract else x) / (jnp.sqrt(var + 1e-12) + 1e-6), stats

def value_targets(phi, theta, data, env):
    """Clipped V-trace(lambda=.96), including reference PG advantage."""
    v, vn = decode_value(critic(phi, data["s"], env)), decode_value(critic(phi, data["ns"], env))
    logp = jax.nn.log_softmax(agent(theta, data["s"], env)["pi"])
    logmu = jax.nn.log_softmax(data["mu"])
    rho = stop(jnp.exp(((logp - logmu) * jax.nn.one_hot(data["a"], A)).sum(-1)))
    rb, discount = jnp.minimum(rho, 1.0), 0.99 * (1.0 - data["done"])
    delta = rb * (data["r"] + discount * vn - v)
    def trace(error, x):
        dt, d, c = x
        error = dt + d * c * error
        return error, error
    _, errors = jax.lax.scan(trace, jnp.zeros_like(v[0]), (delta, discount, 0.96 * rb), reverse=True)
    target = v + errors
    future = jnp.concatenate((target[1:], vn[-1:]), 0)
    adv = rb * (data["r"] + discount * (0.04 * vn + 0.96 * future) - v)
    return v, errors, adv, rho

critic_opt = optax.adam(0.001)

def fit_critic(phi, opt_state, adv_stats, td_stats, theta, data, env):
    v, td, adv, _ = value_targets(phi, theta, data, env)
    _, adv_stats = normalize(adv, adv_stats)
    td_normalized, td_stats = normalize(td, td_stats, subtract=False)
    target = stop(critic(phi, data["s"], env) + td_normalized)
    loss, grad = jax.value_and_grad(lambda p: 0.5 * ((critic(p, data["s"], env) - target) ** 2).mean())(phi)
    updates, opt_state = critic_opt.update(grad, opt_state, phi)
    phi = optax.apply_updates(phi, updates)
    # Critic/optimizer/normalizers are auxiliaries, never an eta gradient path.
    return tree(stop, (phi, opt_state, adv_stats, td_stats)), loss, jnp.sqrt((td * td).mean())

def init_training(key, q, next_policy, batch):
    ka, kc, ks = jax.random.split(key, 3)
    theta, phi = init_agent(ka, q, next_policy), init_critic(kc)
    stats = jnp.zeros(()), jnp.zeros(()), jnp.ones(())
    return theta, theta, phi, critic_opt.init(phi), stats, stats, reset(ks, batch, "catch")

def sample_window(theta, s, key, inner, env="catch"):
    keys = jax.random.split(key, inner + 1)
    s, train = jax.lax.scan(lambda s, k: collect(theta, s, k, env), s, keys[:-1])
    s, valid = collect(theta, s, keys[-1], env, length=4)
    return s, tree(stop, train), tree(stop, valid)

def outer_actor(theta, phi, adv_stats, valid, env):
    _, _, adv, rho = value_targets(phi, theta, valid, env)
    normalized, _ = normalize(adv, adv_stats)
    out = agent(theta, valid["s"], env)
    logp = (jax.nn.log_softmax(out["pi"]) * jax.nn.one_hot(valid["a"], A)).sum(-1)
    def entropy(x):
        return -(softmax(x) * jax.nn.log_softmax(x)).sum(-1).mean()
    pg = -(logp * stop(normalized)).mean()
    loss = pg - 0.01 * entropy(out["pi"]) - 0.001 * (entropy(out["y"]) + entropy(take(out["z"], valid["a"])))
    return loss, (pg, rho.mean(), jnp.sqrt((adv * adv).mean()))

def meta_objective(eta, state, train, valid, env, lr):
    def adapt(carry, data):
        theta, ema, phi, co, ast, tst = carry
        new_theta, ema, reg = update(theta, ema, eta, data, env, lr)
        (phi, co, ast, tst), closs, td = fit_critic(phi, co, ast, tst, theta, data, env)
        return (new_theta, ema, phi, co, ast, tst), (reg, closs, td)
    adapted, (regs, closs, td) = jax.lax.scan(adapt, state[:-1], train)
    theta, _, phi, _, ast, _ = adapted
    loss, (pg, rho, adv) = outer_actor(theta, phi, ast, valid, env)
    logs = dict(
        pg_loss=pg,
        critic_loss=closs.mean(),
        critic_td_rms=td.mean(),
        rho=rho,
        advantage_rms=adv,
        sample_return=valid["r"].sum() / jnp.maximum(1, valid["done"].sum()),
    )
    return loss + regs.mean(), (adapted, logs)

def setup(key, q, next_policy):
    theta = init_agent(key, q, next_policy)
    _, data = collect(theta, reset(key, 2, "catch"), key, "catch")
    o, n = agent(theta, data["s"], "catch"), agent(theta, data["ns"], "catch")
    return init_rule(key, features(o, n, o, n, data).shape[-1])

def save_rule(path, eta):
    np.savez(path, **{k: np.asarray(v) for k, v in eta.items()})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--next-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--meta-steps", type=int, default=1000)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--lifetime", type=int, default=64, help="Agent/critic updates before a fresh lifetime")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-inner", type=int, default=60)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--meta-lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/full"))
    parser.add_argument("--rule", type=Path, help="Frozen .npz rule: evaluate only")
    args = parser.parse_args()
    if (
        min(args.inner, args.batch, args.eval_inner, args.eval_seeds) < 1
        or args.meta_steps < 0
        or args.lifetime < args.inner
    ):
        parser.error("counts must be positive (meta-steps may be zero)")
    args.out.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    eta0 = setup(key, args.q, args.next_policy)
    eta = eta0 if args.rule is None else {k: jnp.asarray(v) for k, v in np.load(args.rule).items()}
    if eta["wx"].shape != eta0["wx"].shape:
        parser.error("rule feature shape differs: use its original --q/--no-q setting")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.meta_lr))
    opt_state = optimizer.init(eta)
    @jax.jit
    def meta_step(eta, opt_state, state, key):
        s, train, valid = sample_window(state[0], state[-1], key, args.inner)
        (loss, (adapted, logs)), grad = jax.value_and_grad(meta_objective, has_aux=True)(
            eta, state, train, valid, "catch", args.lr
        )
        updates, opt_state = optimizer.update(grad, opt_state, eta)
        return (
            optax.apply_updates(eta, updates),
            opt_state,
            tree(stop, adapted + (s,)),
            loss,
            optax.tree.norm(grad),
            logs,
        )
    start, logs = time.monotonic(), []
    state = init_training(key, args.q, args.next_policy, args.batch)
    for i in range(0 if args.rule else args.meta_steps):
        key, sub = jax.random.split(key)
        if i % max(1, args.lifetime // args.inner) == 0:
            key, fresh = jax.random.split(key)
            state = init_training(fresh, args.q, args.next_policy, args.batch)
        eta, opt_state, state, loss, norm, metrics = meta_step(eta, opt_state, state, sub)
        if i % 10 == 0 or i + 1 == args.meta_steps:
            row = dict(step=i + 1, objective=float(loss), grad_norm=float(norm), seconds=time.monotonic() - start)
            row.update({k: float(v) for k, v in metrics.items()})
            logs.append(row)
            print(json.dumps(row), flush=True)
    save_rule(args.out / "rule.npz", eta)
    save_rule(args.out / "initial_rule.npz", eta0)
    if args.rule is None and args.meta_steps:
        save_rule(args.out / "meta_critic.npz", state[2])
    np.savez(
        args.out / "meta_curve.npz",
        **{k: [r[k] for r in logs] for k in (logs[0] if logs else ("step", "objective", "grad_norm", "seconds"))},
    )
    summary = dict(
        mode="evaluation_only" if args.rule else "meta_training",
        training_objective="learned_meta_critic_vtrace",
        grid=[STEPS + 1, WIDTH, 2],
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        devices=[str(d) for d in jax.devices()],
        seconds=time.monotonic() - start,
        environments={},
    )
    # Deliberate module boundary: exact model enumeration is evaluation only.
    from evaluate import exact_return, lifetime
    for env in ("catch", "reach"):
        evaluate = jax.jit(
            lambda rule, k: lifetime(rule, k, env, args.q, args.next_policy, args.eval_inner, args.batch, args.lr)[1]
        )
        keys = [jax.random.PRNGKey(10000 + s) for s in range(args.eval_seeds)]
        curves = np.stack([np.asarray(evaluate(eta, k)) for k in keys])
        initial = np.stack([np.asarray(evaluate(eta0, k)) for k in keys])
        before = [
            float(exact_return(init_agent(jax.random.split(k, 3)[0], args.q, args.next_policy), env)) for k in keys
        ]
        np.savez(args.out / f"{env}_curves.npz", learned=curves, untrained=initial, before=before)
        summary["environments"][env] = dict(
            before=float(np.mean(before)),
            learned_final=float(curves[:, -1].mean()),
            untrained_final=float(initial[:, -1].mean()),
            seed_finals=curves[:, -1].tolist(),
        )
        print(env, json.dumps(summary["environments"][env]), flush=True)
    summary["seconds"] = time.monotonic() - start
    summary["finite"] = all(np.isfinite(e["seed_finals"]).all() for e in summary["environments"].values())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__":
    main()
