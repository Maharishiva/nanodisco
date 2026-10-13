"""Issue 1 check: does the meta-objective's data timing matter?

usage: python scripts/outer_check.py OUT pre|post [--seeds 0 1 2 3] [--steps 800] [--eval-steps 800]
pre:  mini.train as is: the validation rollout is collected before the inner updates (by mu = pi_theta),
      then the adapted policy pi_theta' is scored on it with no importance correction (as DiscoRL's notebook).
post: the validation rollout is collected after the inner updates, by the adapted agent, so the policy
      gradient is on-policy for pi_theta'.
Both log per meta-step the training return and how far the adapted policy moved on the validation data:
KL(mu || pi_theta') and mean |pi_theta'(a)/mu(a) - 1| over the taken actions. Then each final rule trains
fresh agents frozen (mini.run_frozen), as in scripts/catch_frozen.py. Rules go to OUT/VARIANT/ruleSEED.npz.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "history"))
import mini_v2 as m  # written for the on-policy mini v2

def drift(outputs, valid):  # how far the adapted policy is from the one that collected valid
    logp, logmu = m.lsm(outputs["pi"]), m.lsm(valid["mu"])
    kl = (jnp.exp(logmu) * (logmu - logp)).sum(-1).mean()
    ratio = jnp.exp(m.pick(logp - logmu, valid["a"])[..., 0])
    return jnp.stack([kl, jnp.abs(ratio - 1).mean()])

def score(params, valid, critic):  # policy-gradient meta-loss of the adapted agent on valid, as mini.meta_loss
    outputs = m.drop_last(m.unroll(params, valid, "catch"))
    v, returns = m.critic_targets(critic, valid)
    logp = m.pick(m.lsm(outputs["pi"]), valid["a"])[..., 0]
    loss = -(logp * m.stop(m.norm(returns - v))).mean() - 1e-2 * m.entropy(outputs["pi"])
    loss -= 1e-3 * (m.entropy(outputs["y"]) + m.entropy(m.take(outputs["z"], valid["a"])))
    return loss, m.stop(drift(outputs, valid))

def meta_loss(eta, params, opt_state, train, carry, key_valid, critic, post):
    def adapt(state, rollout):
        params, opt_state, reg = m.inner_step(*state, eta, rollout, "catch")
        return (params, opt_state), reg
    params0 = params
    (params, opt_state), regs = jax.lax.scan(adapt, (params, opt_state), train)
    actor = m.stop(params) if post else params0  # who collects the validation rollout
    carry, valid = m.collect(actor, carry, key_valid, "catch", 2 * m.ROLL)
    loss, diag = score(params, valid, critic)
    return loss + regs.mean(), (params, opt_state, carry, valid, diag)

def train(key, steps, post):
    def one_agent(eta, agent, key):
        params, opt_state, critic, critic_opt_state, carry = agent
        key_train, key_valid = jax.random.split(key)
        def one(carry, key):
            return m.collect(params, carry, key, "catch")
        carry, train = jax.lax.scan(one, carry, jax.random.split(key_train, m.INNER))
        def fit(state, rollout):
            return m.fit_critic(*state, rollout), None
        critic, critic_opt_state = jax.lax.scan(fit, (critic, critic_opt_state), train)[0]
        grad_fn = jax.value_and_grad(meta_loss, has_aux=True)
        (loss, aux), grads = grad_fn(eta, params, opt_state, train, carry, key_valid, critic, post)
        params, opt_state, carry, valid, diag = aux
        ret = m.episode_return(train, valid)
        return grads, (params, opt_state, critic, critic_opt_state, carry), (loss, ret, diag)
    def meta_step(state, key):
        eta, meta_opt_state, agents = state
        keys = jax.random.split(key, m.AGENTS)
        grads, agents, (loss, ret, diag) = jax.vmap(one_agent, (None, 0, 0))(eta, agents, keys)
        updates, meta_opt_state = m.meta_optimizer.update(jax.tree.map(lambda g: g.mean(0), grads), meta_opt_state, eta)
        logs = dict(meta_loss=loss.mean(), train_return=ret.mean(), kl=diag[:, 0].mean(), ratio_dev=diag[:, 1].mean())
        return (optax.apply_updates(eta, updates), meta_opt_state, agents), logs
    key_eta, key_agents, key_steps = jax.random.split(key, 3)
    eta, agents = m.init_rule(key_eta), jax.vmap(m.init_agent_state)(jax.random.split(key_agents, m.AGENTS))
    state = (eta, m.meta_optimizer.init(eta), agents)
    (eta, _, _), logs = jax.lax.scan(meta_step, state, jax.random.split(key_steps, steps))
    return eta, logs

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", type=Path)
p.add_argument("variant", choices=("pre", "post"))
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
p.add_argument("--steps", type=int, default=800)
p.add_argument("--eval-steps", type=int, default=800)
a = p.parse_args()
out = a.out / a.variant
out.mkdir(parents=True, exist_ok=True)
train_jit = jax.jit(lambda k: train(k, a.steps, a.variant == "post"))
frozen = jax.jit(lambda e, k: m.run_frozen(e, k, a.eval_steps))
summary = {}
for seed in a.seeds:
    start = time.monotonic()
    eta, logs = jax.block_until_ready(train_jit(jax.random.PRNGKey(seed)))
    fresh = np.asarray(frozen(eta, jax.random.PRNGKey(1000 + seed)))
    logs = {k: np.asarray(v) for k, v in logs.items()}
    np.savez(out / f"seed{seed}.npz", fresh=fresh, **logs)
    rule = {jax.tree_util.keystr(p): np.asarray(x) for p, x in jax.tree_util.tree_leaves_with_path(eta)}
    np.savez(out / f"rule{seed}.npz", **rule)  # bench.load_rule format
    w = len(fresh) // 10
    summary[seed] = dict(
        seconds=time.monotonic() - start,
        meta_training_last=float(logs["train_return"][-80:].mean()),
        kl_mean=float(logs["kl"].mean()),
        ratio_dev_mean=float(logs["ratio_dev"].mean()),
        fresh_windows=[float(fresh[i : i + w].mean()) for i in range(0, len(fresh), w)],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    s = summary[seed]
    print(f"{a.variant} seed {seed}: meta-train last {s['meta_training_last']:+.2f}  KL(mu||pi') {s['kl_mean']:.4f}  "
          f"|ratio-1| {s['ratio_dev_mean']:.3f}  fresh agents " + " ".join(f"{x:+.2f}" for x in s["fresh_windows"]),
          flush=True)
