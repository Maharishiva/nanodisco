"""Evaluation-only exact MDP enumeration. Never used to optimize eta."""

from disco import *

def states():
    return jnp.stack(jnp.meshgrid(jnp.arange(STEPS), jnp.arange(WIDTH), jnp.arange(WIDTH), indexing="ij"), -1).reshape(
        -1, 3
    )

def exact_return(theta, env):
    """Exact return for evaluation ONLY; never imported by meta-training."""
    ss = states()
    pi = stop(softmax(agent(theta, ss, env)["pi"]))
    ns, rewards, done = transition(ss[:, None, :], jnp.arange(A)[None, :], env)
    indices = (ns[..., 0] * WIDTH + ns[..., 1]) * WIDTH + ns[..., 2]
    initial = (ss[:, 0] == 0) & (ss[:, 2] == WIDTH // 2)
    if env == "reach":
        initial &= (ss[:, 1] == 0) | (ss[:, 1] == WIDTH - 1)
    mass = initial.astype(jnp.float32) / initial.sum()
    def step(carry, _):
        mass, ret = carry
        flow = mass[:, None] * pi
        ret += (flow * rewards).sum()
        mass = jnp.zeros_like(mass).at[indices].add(flow * (1.0 - done))
        return (mass, ret), None
    return stop(jax.lax.scan(step, (mass, 0.0), None, length=STEPS)[0][1])

def lifetime(eta, key, env, q, next_policy, inner=12, batch=32, lr=2.0):
    ka, ks, key = jax.random.split(key, 3)
    theta = init_agent(ka, q, next_policy)
    def step(carry, key):
        th, ema, s = carry
        s, data = collect(th, s, key, env)
        th, ema, reg = update(th, ema, eta, data, env, lr)
        return (th, ema, s), (exact_return(th, env), reg)
    (theta, _, _), (curve, reg) = jax.lax.scan(
        step, (theta, theta, reset(ks, batch, env)), jax.random.split(key, inner)
    )
    return theta, curve, reg.mean()
