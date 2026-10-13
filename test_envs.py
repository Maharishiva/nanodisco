"""Checks for mini.py's env-module hook; run: python -m unittest test_envs -v.

A pure-JAX toy env module with packed frames and two games (game 1 forbids action 2) stands in for the ARC engine.
"""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import mini as m

SMALL = dict(INNER=2, VALID=1, BATCH=8, BUFFER=29 * 32, AGENTS=2)

def toy_reset(key, game, n):
    zero = jnp.zeros(n, jnp.int32)
    return dict(t=zero, x=zero + 2, game=zero + game)

def toy_step(key, state, action):  # move left/stay/right for 4 steps, +1 when ending in the middle
    x, t = jnp.clip(state["x"] + action - 1, 0, 4), state["t"] + 1
    done = t == 4
    new = dict(t=jnp.where(done, 0, t), x=jnp.where(done, 2, x), game=state["game"])
    return new, 1.0 * done * (x == 2), done

toy = types.SimpleNamespace(
    OBS=2048, A=3, FRAME=True, GAMES=("free", "no_right"), reset=toy_reset, step=toy_step,
    obs=lambda s: (17 * (jnp.arange(2048) // 400 == s["x"][:, None])).astype(jnp.uint8),
    valid=lambda s: jnp.array([[True, True, True], [True, True, False]])[s["game"]],
)

class EnvHookTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(m, k) for k in list(SMALL) + ["ENV", "OBS", "A"]}
        for k, v in SMALL.items():
            setattr(m, k, v)
        m.use_env("toy", toy)
    def tearDown(self):
        for k, v in self.saved.items():
            setattr(m, k, v)
        m.ENVS.pop("toy")
    def test_rollout_records_frames_and_legal_actions(self):
        key = jax.random.PRNGKey(0)
        agent = m.init_agent_state(key, "toy", 1)
        self.assertIn("stem", agent[0])
        self.assertIn("stem", agent[2])
        rollout = jax.jit(lambda a, c: m.collect(a, c, key, "toy", 64))(agent[0], agent[4])[1]
        self.assertEqual(rollout["s"].shape, (65, m.B, 2048))
        self.assertEqual(rollout["s"].dtype, jnp.uint8)
        self.assertEqual(rollout["valid"].shape, (65, m.B, 3))
        self.assertFalse(bool((rollout["a"] == 2).any()))  # game 1 never plays its illegal action
        self.assertTrue(bool((rollout["a"] == 0).any()))
        out = m.unroll(agent[0], rollout, "toy")
        self.assertTrue(bool((out["pi"][..., 2] < -1e8).all()))
    def test_meta_training_and_frozen_rule_on_frames(self):
        np.testing.assert_array_equal(m.init_agents(jax.random.PRNGKey(0), 2, "toy")[4][0]["game"][:, 0], [0, 1])
        eta, state, logs = jax.jit(lambda k: m.train(k, 3, lifetimes=(2,)))(jax.random.PRNGKey(1))
        self.assertTrue(all(np.isfinite(v).all() for v in logs.values()))
        games = np.asarray(state[2][4][0]["game"])  # restarted agents drew a game; all envs of an agent share it
        self.assertTrue(((games >= 0) & (games < 2) & (games == games[:, :1])).all())
        frozen = jax.jit(lambda e, k: m.run_frozen(e, k, 4, "toy"))(eta, jax.random.PRNGKey(2))
        self.assertTrue(np.isfinite(frozen).all())
    def test_stem_reads_patches(self):
        frame = jnp.zeros((1, 2048), jnp.uint8).at[0, 0].set(0x21)  # pixel (0, 0) colour 1, pixel (1, 0) colour 2
        p = m.init_stem(jax.random.PRNGKey(3))
        self.assertEqual(m.stem(p, frame).shape, (1, m.STEM))
        g = jax.grad(lambda p: m.stem(p, frame).sum())(p)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(g)))

if __name__ == "__main__":
    unittest.main()
