"""Small semantic tests; run: python -m unittest -v."""

import itertools
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import disco as d
import evaluate as e

class DiscoTests(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(7)
    def test_pixels_and_different_tasks(self):
        s = jnp.array([[0, 0, 1]])
        self.assertEqual(d.pixels(s, "catch").shape, (1, d.STEPS + 1, d.WIDTH, 2))
        self.assertFalse(np.array_equal(d.pixels(s, "catch"), d.pixels(s, "reach")))
        _, rc, dc = d.transition(s, jnp.array([0]), "catch")
        _, rr, dr = d.transition(s, jnp.array([0]), "reach")
        self.assertEqual(float(rc[0]), 0.0)
        self.assertFalse(bool(dc[0]))
        self.assertEqual(float(rr[0]), 1.0)
        self.assertTrue(bool(dr[0]))
    def test_exact_return_matches_bruteforce(self):
        theta = d.init_agent(self.key)
        for env in ("catch", "reach"):
            total = 0.0
            goals = range(d.WIDTH) if env == "catch" else (0, d.WIDTH - 1)
            pi = np.array(d.softmax(d.agent(theta, e.states(), env)["pi"])).reshape(d.STEPS, d.WIDTH, d.WIDTH, 3)
            for goal in goals:
                for actions in itertools.product(range(3), repeat=d.STEPS):
                    player, prob, reward_sum, ended = d.WIDTH // 2, 1.0, 0.0, False
                    for t, a in enumerate(actions):
                        if ended:  # Remaining actions form a normalized dummy distribution.
                            prob /= 3
                            continue
                        prob *= pi[t, goal, player, a]
                        player = int(np.clip(player + a - 1, 0, d.WIDTH - 1))
                        hit, timeout = player == goal, t == d.STEPS - 1
                        reward_sum += (
                            (float(2 * hit - 1) if timeout else 0.0) if env == "catch" else (1.0 if hit else 0.0)
                        )
                        ended = timeout if env == "catch" else hit or timeout
                    total += prob * reward_sum / len(goals)
            self.assertAlmostEqual(float(e.exact_return(theta, env)), total, places=5)
    def test_switches_remove_heads_and_q_features(self):
        sizes = {}
        for q, nxt in itertools.product((False, True), repeat=2):
            theta = d.init_agent(self.key, q, nxt)
            self.assertEqual("q" in theta, q)
            self.assertEqual("next" in theta, nxt)
            sizes[q, nxt] = d.setup(self.key, q, nxt)["wx"].shape[0]
        self.assertEqual(sizes[False, False], sizes[False, True])
        self.assertEqual(sizes[True, False], sizes[True, True])
        self.assertEqual(sizes[True, True] - sizes[False, False], 3)
    def test_kl_teacher_direction(self):
        teacher, student = jnp.array([3.0, 0.0, -2.0]), jnp.zeros(3)
        g = jax.grad(lambda x: d.kl(teacher, x))(student)
        np.testing.assert_allclose(g, d.softmax(student) - d.softmax(teacher), atol=1e-6)
        self.assertNotAlmostEqual(float(d.kl(teacher, student)), float(d.kl(student, teacher)))
    def test_no_q_bootstrap_inputs_and_meta_gradient(self):
        theta = d.init_agent(self.key, False, False)
        eta = d.setup(self.key, False, False)
        s, data = d.collect(theta, d.reset(self.key, 8, "catch"), self.key, "catch", length=4)
        o, n = d.agent(theta, data["s"], "catch"), d.agent(theta, data["ns"], "catch")
        def prediction(y):
            changed = dict(n, y=y)
            x = d.features(o, changed, o, changed, data)
            return d.softmax(d.rule(eta, x, data["done"])[0])[..., 0].sum()
        self.assertGreater(float(jnp.linalg.norm(jax.grad(prediction)(n["y"]))), 1e-8)
        _, valid = d.collect(theta, s, jax.random.PRNGKey(91), "catch", length=4)
        phi = d.init_critic(self.key)
        _, _, advantages, _ = d.value_targets(phi, theta, valid, "catch")
        frozen_adv = d.stop(advantages)
        def outer(rule):
            th, _, _ = d.update(theta, theta, rule, data, "catch", 2.0)
            logp = jax.nn.log_softmax(d.agent(th, valid["s"], "catch")["pi"])
            return -(logp * jax.nn.one_hot(valid["a"], d.A) * frozen_adv[..., None]).sum(-1).mean()
        grad = jax.grad(outer)(eta)
        self.assertGreater(float(jnp.linalg.norm(grad["out"])), 1e-7)
        # Mixed derivative through the inner KL update, checked numerically.
        i = np.unravel_index(np.abs(grad["out"]).argmax(), grad["out"].shape)
        eps = 0.003
        plus, minus = dict(eta), dict(eta)
        plus["out"] = eta["out"].at[i].add(eps)
        minus["out"] = eta["out"].at[i].add(-eps)
        numeric = float((outer(plus) - outer(minus)) / (2 * eps))
        self.assertAlmostEqual(float(grad["out"][i]), numeric, delta=3e-5)
    def test_terminal_reset_and_frozen_transfer(self):
        eta = d.setup(self.key, False, False)
        x = jax.random.normal(self.key, (2, 1, 3, eta["wx"].shape[0]))
        done = jnp.array([[True], [False]])
        first = d.rule(eta, x, done)[0][0]
        changed = d.rule(eta, x.at[1].set(100.0), done)[0][0]
        np.testing.assert_allclose(first, changed, atol=1e-7)
        before = {k: np.array(v) for k, v in eta.items()}
        _, curve, _ = e.lifetime(eta, self.key, "reach", False, False, inner=2, batch=3)
        self.assertTrue(np.isfinite(curve).all())
        for k in eta:
            np.testing.assert_array_equal(before[k], eta[k])
    def test_sparse_reach_rewards(self):
        s = jnp.array([[0, 0, 2], [3, 0, 2]])
        _, rewards, done = d.transition(s, jnp.array([1, 1]), "reach")
        np.testing.assert_array_equal(rewards, [0.0, 0.0])
        np.testing.assert_array_equal(done, [False, True])
    def test_vtrace_targets_and_terminal_mask(self):
        theta = d.init_agent(self.key, False, False)
        theta["pi"] = jnp.zeros_like(theta["pi"])
        phi = d.tree(jnp.zeros_like, d.init_critic(self.key))
        data = dict(
            s=jnp.array([[[2, 2, 2]], [[3, 2, 2]]]),
            ns=jnp.array([[[3, 2, 2]], [[3, 2, 2]]]),
            a=jnp.ones((2, 1), jnp.int32),
            r=jnp.array([[0.0], [1.0]]),
            done=jnp.array([[False], [True]]),
            mu=jnp.zeros((2, 1, 3)),
        )
        _, td, adv, rho = d.value_targets(phi, theta, data, "catch")
        np.testing.assert_allclose(td[:, 0], [0.99 * 0.96, 1.0], atol=1e-6)
        np.testing.assert_allclose(adv, td, atol=1e-6)
        np.testing.assert_allclose(rho, 1.0, atol=1e-6)
        data["mu"] = data["mu"].at[..., 1].set(jnp.log(2.0))
        _, td, _, rho = d.value_targets(phi, theta, data, "catch")
        np.testing.assert_allclose(rho, 2 / 3, atol=1e-6)
        np.testing.assert_allclose(td[:, 0], [0.99 * 0.96 * 4 / 9, 2 / 3], atol=1e-6)
    def test_real_critic_fit_and_stopped_actor_advantages(self):
        state = d.init_training(self.key, False, False, 8)
        th, _, phi, co, ast, tst, s = state
        _, data = d.collect(th, s, self.key, "catch", length=8)
        (new_phi, _, _, _), loss, _ = d.fit_critic(phi, co, ast, tst, th, data, "catch")
        delta = float(sum(jnp.sum((new_phi[k] - phi[k]) ** 2) for k in phi))
        self.assertGreater(delta, 1e-8)
        self.assertTrue(np.isfinite(loss))
        gphi = jax.grad(lambda p: d.outer_actor(th, p, ast, data, "catch")[0])(phi)
        self.assertEqual(float(sum(jnp.sum(jnp.abs(v)) for v in gphi.values())), 0.0)
    def test_heldout_fit_isolation_and_no_oracle_gradient(self):
        state = d.init_training(self.key, False, False, 8)
        eta = d.setup(self.key, False, False)
        _, train, valid = d.sample_window(state[0], state[-1], self.key, 2)
        with patch.object(e, "exact_return", side_effect=AssertionError("oracle in training")):
            (loss, (adapted, _)), grad = jax.value_and_grad(d.meta_objective, has_aux=True)(
                eta, state, train, valid, "catch", 2.0
            )
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(jnp.linalg.norm(grad["out"])), 1e-9)
        other = dict(valid, r=valid["r"] + 10.0)
        _, (second, _) = d.meta_objective(eta, state, train, other, "catch", 2.0)
        for a, b in zip(jax.tree.leaves(adapted[2:]), jax.tree.leaves(second[2:])):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)  # Allow AD compiler roundoff only.
        oracle_grad = jax.grad(lambda th: e.exact_return(th, "catch"))(state[0])
        self.assertEqual(float(sum(jnp.sum(jnp.abs(v)) for v in oracle_grad.values())), 0.0)

if __name__ == "__main__":
    unittest.main()
