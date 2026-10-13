"""How a frozen rule's policy target changes with the number of actions, on synthetic rollouts (inference only).

usage: python scripts/rule_actions_probe.py NAME=RULE.npz[:retrace] ...
Cases: Q flat (no evidence) or one action better by GAP, the agent's policy uniform, peaked (random logits, std 1.5),
or peaked on a worse action. Printed per case and number of actions A: the entropy of the target and of the policy over
log A, the gain (target's log-probability gap between the better action and the mean of the others, minus the policy's),
and KL(target || policy). A rule that ignores the policy shows a target entropy near 1 when Q is flat and the policy is
peaked: it pulls every policy back to near uniform (the rules meta-trained without RULE_LOGP do this).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import mini as m

T, N, NS = 29, 96, (3, 6, 17)
CASES = (("flat q, pi uniform", 0.0, 0.0, False), ("flat q, pi peaked", 1.5, 0.0, False),
         ("q gap 0.5, pi uniform", 0.0, 0.5, False), ("q gap 0.05, pi uniform", 0.0, 0.05, False),
         ("q gap 0.5, pi on a worse action", 0.0, 0.5, True))

def probe(eta, n_actions, logit_std, gap, worse, key):
    m.A = n_actions
    k = jax.random.split(key, 6)
    good = jax.random.randint(k[1], (T + 1, N), 0, n_actions)
    pi = logit_std * jax.random.normal(k[0], (T + 1, N, n_actions))
    if worse:  # the policy prefers the action after the better one
        pi = 2.0 * jax.nn.one_hot((good + 1) % n_actions, n_actions) + 0.3 * jax.random.normal(k[0], pi.shape)
    q = gap * jax.nn.one_hot(good, n_actions) + 0.02 * jax.random.normal(k[2], (T + 1, N, n_actions))
    a = jax.random.categorical(k[3], pi[:-1])
    r = 1.0 * (jax.random.uniform(k[4], (T, N)) < 0.02)
    done = 1.0 * (jax.random.uniform(k[5], (T, N)) < 0.01)
    outputs = dict(pi=pi, q=q, y=jnp.zeros((T + 1, N, m.D)), z=jnp.zeros((T + 1, N, n_actions, m.D)))
    rollout = dict(a=a, r=r, done=done, mu=pi[:-1], first=jnp.zeros((T + 1, N)))
    values = m.q_stats(outputs, rollout)
    target = m.anchor(m.rule(eta, *m.features(outputs, rollout, values), rollout), outputs)[0]
    is_good = jax.nn.one_hot(good[:-1], n_actions)
    def gap_of(logits):
        logp = jax.nn.log_softmax(logits)
        return ((logp * is_good).sum(-1) - (logp * (1 - is_good)).sum(-1) / (n_actions - 1)).mean()
    log_n = np.log(n_actions)
    return (float(m.entropy(target) / log_n), float(m.entropy(pi[:-1]) / log_n),
            float(gap_of(target) - gap_of(pi[:-1])), float(m.kl(target, pi[:-1]).mean()))

m.USE_Q = True
for arg in sys.argv[1:]:
    name, path = arg.split("=")
    path, _, flag = path.partition(":")
    m.RETRACE = flag == "retrace"
    eta = bench.load_rule(path)
    print(f"{name} (RULE_LOGP {m.RULE_LOGP}, RULE_LEGAL {m.RULE_LEGAL}, ANCHOR {m.ANCHOR})")
    for label, std, gap, worse in CASES:
        row = [probe(eta, n, std, gap, worse, jax.random.PRNGKey(n)) for n in NS]
        print(f"  {label:32s}" + " | ".join(f"A={n:2d} H target {e:.2f} pi {h:.2f} gain {g:+.2f} KL {kl:.2f}"
                                             for n, (e, h, g, kl) in zip(NS, row)), flush=True)
