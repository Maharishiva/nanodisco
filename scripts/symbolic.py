"""A three-term symbolic stand-in for the learned value-input rule, fitted to its updates on Catch.

target logits_a = 0.57 log pi_a + 0.83 norm(Q_a - V) + 0.98 [a = a_t]
norm() is over the whole batch, as in the rule's input. The agent minimizes KL(target || pi) plus the Q head's
fixed TD(lambda) loss; there are no y/z targets. step() is a drop-in for mini.inner_step(...)[:2].
"""

import jax

import mini as m

COEF = (0.57, 0.83, 0.98)  # log pi, norm(Q - V), taken action

def loss(params, targets, rollout, env):  # KL to the formula's policy target + the Q head's TD loss
    outputs = m.drop_last(m.unroll(params, rollout, env))
    a = rollout["a"]
    return (m.kl(targets[0], outputs["pi"]) + 0.5 * (m.pick(outputs["q"], a)[..., 0] - targets[1]) ** 2).mean()

def step(params, opt_state, rollout, env, coef=COEF):
    outputs = m.unroll(params, rollout, env)
    values = m.q_stats(outputs, rollout)
    target = (
        coef[0] * m.lsm(m.stop(outputs["pi"][:-1]))
        + coef[1] * m.norm(values["qv"][:-1])
        + coef[2] * jax.nn.one_hot(rollout["a"], m.A)
    )
    return m.adam(params, opt_state, jax.grad(loss)(params, (target, values["g"]), rollout, env))
