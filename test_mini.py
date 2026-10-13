"""Checks for mini.py; run: python -m unittest test_mini -v. Small settings keep it fast."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mini as m

SMALL = dict(INNER=2, VALID=1, BATCH=8, BUFFER=29 * 32)
CORES = (("gru", "gru"), ("mlp", "attn"), ("gdn", "gdn"))  # (AGENT_CORE, RULE_CORE)
TINY_CNN = dict(AGENT_CORE="impala", PIXELS=(9, 9, 3), CNN=(4, 8, 8), CNN_FC=16, A=5, OBS=9 * 9 * 3 + 1)

class MiniTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(m, k) for k in (*SMALL, *TINY_CNN, "RULE_CORE", "CNN_DTYPE", "ACT_MAX", "GAME_A",
                                                 "ACT_REDRAW", "RULE_LOGP", "RULE_LEGAL", "ANCHOR", "LOGP_GRAD",
                                                 "ACT_PLAIN", "UTD", "REMAT", "CNN_REMAT", "CNN_STEPS")}
        for k, v in SMALL.items():
            setattr(m, k, v)
        self.build()
    def build(self):
        k = jax.random.PRNGKey(3)
        self.key = jax.random.fold_in(k, 2)
        self.eta = m.init_rule(jax.random.fold_in(k, 1))
        self.agent = m.init_agent_state(k)
        self.rollout = m.collect(self.agent[0], self.agent[4], jax.random.fold_in(k, 3), "catch")[1]
    def tearDown(self):
        for k, v in self.saved.items():
            setattr(m, k, v)
        m.ENVS.pop("pixels", None)
    def meta_grad(self, mixflow):
        def f(e):
            return m.meta_loss(e, self.agent, self.key, mixflow)[0]
        with jax.default_matmul_precision("highest"):  # GPU TF32 rounding differs between the two orders by ~1%
            return jax.jit(jax.grad(f))(self.eta)
    def test_mixflow_matches_reverse_over_reverse(self):
        for cores in CORES:
            with self.subTest(cores=cores):
                m.AGENT_CORE, m.RULE_CORE = cores
                self.build()
                mixed, plain = self.meta_grad(True), self.meta_grad(False)
                self.assertGreater(float(m.optax.tree.norm(plain)), 1e-6)
                for a, b in zip(jax.tree.leaves(mixed), jax.tree.leaves(plain)):  # float64: equal to ~1e-11, all cores
                    np.testing.assert_allclose(a, b, rtol=1e-3, atol=max(1e-6, 3e-3 * float(np.abs(b).max())))
    def test_remat_keeps_the_meta_gradient(self):  # REMAT recomputes the agent updates in the backward pass
        stored = self.meta_grad(True)
        m.REMAT = True
        for a, b in zip(jax.tree.leaves(self.meta_grad(True)), jax.tree.leaves(stored)):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3 * float(np.abs(b).max()))
    def test_fit_critic_is_the_td_gradient(self):  # one shared forward pass, the same Adam step as the plain loss
        critic, opt_state = self.agent[2], self.agent[3]
        logits = jax.random.normal(self.key, self.rollout["mu"].shape)
        new = m.fit_critic(critic, opt_state, self.rollout, logits)[0]
        targets = m.critic_targets(critic, self.rollout, logits)[0]
        def loss(c):
            return 0.5 * ((m.critic_values(c, self.rollout)[:-1] - targets) ** 2).mean()
        updates = m.critic_optimizer.update(jax.grad(loss)(critic), opt_state)[0]
        for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(m.optax.apply_updates(critic, updates))):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    def test_max_pool_is_reduce_window(self):  # values and gradients, also with ties (integer inputs)
        for shape in ((2, 9, 9, 3), (1, 8, 7, 2), (2, 63, 63, 4)):
            x = jnp.round(4 * jax.random.normal(jax.random.PRNGKey(sum(shape)), shape))
            def pool(x):
                return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME")
            probe = jax.random.normal(jax.random.PRNGKey(1), pool(x).shape)
            for f in (lambda f: f(x), lambda f: jax.grad(lambda x: (f(x) * probe).sum())(x)):
                np.testing.assert_array_equal(f(m.max_pool), f(pool))
    def test_gdn_chunks_match_steps(self):  # chunkwise form = one step at a time, with a start state and resets
        k = jax.random.split(jax.random.PRNGKey(9), 4)
        layer, x = m.init_gdn(k[0], 64, 2), jax.random.normal(k[1], (30, 3, 64))
        first = 1.0 * (jax.random.uniform(k[2], (30, 3)) < 0.15)
        S = 0.3 * jax.random.normal(k[3], (3, 2, m.GDN_K, m.GDN_K))
        def step(S, inputs):
            x, f = inputs
            return m.gdn_step(layer, S * (1 - f)[:, None, None, None], x)
        with jax.default_matmul_precision("highest"):
            np.testing.assert_allclose(m.gdn_seq(layer, S, x, first), jax.lax.scan(step, S, (x, first))[1], atol=1e-5)
    def test_assoc_scans_match_loops(self):
        k = jax.random.split(jax.random.PRNGKey(10), 6)
        r, v, q, c = (jax.random.normal(x, (29, 5)) for x in k[:4])
        discount, c = 0.99 * (jax.random.uniform(k[4], (29, 5)) > 0.2), 0.5 * jnp.abs(c)
        def all_three():
            return m.lambda_returns(r, discount, v, 0.9), m.retrace(r, discount, v, q, c), m.vtrace(
                jnp.concatenate([v, v[:1]]), r, discount, c, 0.9)
        loops, m.ASSOC = all_three(), True
        try:
            for a, b in zip(jax.tree.leaves(loops), jax.tree.leaves(all_three())):
                np.testing.assert_allclose(a, b, atol=1e-5)
        finally:
            m.ASSOC = False
    def test_rule_attention_sees_only_later_steps_of_its_episode(self):
        m.RULE_CORE = "attn"
        layer = m.init_blocks(jax.random.PRNGKey(11), 10)["blocks"][0]["mix"]
        x = jax.random.normal(jax.random.PRNGKey(12), (12, 1, m.RULE_DIM))
        done = jnp.zeros((12, 1)).at[8].set(1.0)  # the episode of step 3 is steps 0..8
        out = m.attend(layer, x, done)
        outside = m.attend(layer, x.at[:3].add(1.0).at[9:].add(1.0), done)
        np.testing.assert_allclose(outside[3], out[3], atol=1e-6)
        self.assertGreater(float(jnp.abs(m.attend(layer, x.at[8].add(1.0), done)[3] - out[3]).max()), 1e-4)
    def test_mixflow_forward_pass_is_plain(self):
        a = m.inner_step(self.agent[0], self.agent[1], self.eta, self.rollout, "catch", True)[0]
        b = m.inner_step(self.agent[0], self.agent[1], self.eta, self.rollout, "catch", False)[0]
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)
    def test_frozen_step_is_inner_step(self):  # one shared forward pass gives the same gradient
        adam = m.adam
        m.adam = lambda params, opt_state, grads: (grads, opt_state)  # compare the gradients, not Adam's signs
        try:
            for cores in CORES[:2] + (("impala", "gru"),):
                with self.subTest(cores=cores):
                    m.AGENT_CORE, m.RULE_CORE = cores
                    d, env = self.rollout, "catch"
                    if cores[0] == "impala":
                        for k, v in TINY_CNN.items():
                            setattr(m, k, v)
                        m.ENVS["pixels"], m.CNN_DTYPE, env = types.SimpleNamespace(), jnp.float32, "pixels"
                        k = jax.random.split(jax.random.PRNGKey(14), 2)
                        frames = jax.random.randint(k[0], d["first"].shape + (m.OBS,), 0, m.A).astype(jnp.uint8)
                        d = d | dict(s=frames, mu=jax.random.normal(k[1], d["a"].shape + (m.A,)),
                                     h0=jnp.zeros(d["h0"].shape[:1] + (m.memory_size(),)))
                    params, eta = m.init_agent(jax.random.PRNGKey(15)), m.init_rule(jax.random.PRNGKey(16))
                    opt = (params, params, jnp.zeros(()))
                    with jax.default_matmul_precision("highest"):
                        a = jax.jit(lambda p: m.inner_step(p, opt, eta, d, env)[0])(params)
                        b = jax.jit(lambda p: m.frozen_step(p, opt, eta, d, env)[0])(params)
                    self.assertGreater(float(m.optax.tree.norm(a)), 1e-6)
                    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6 * float(np.abs(x).max()) + 1e-9)
        finally:
            m.adam = adam
    def test_impala_core(self):  # unroll = agent_step over time; MixFlow = plain meta-gradient through an update
        for k, v in TINY_CNN.items():
            setattr(m, k, v)
        m.ENVS["pixels"] = types.SimpleNamespace()  # rollouts of env modules hold observations
        k = jax.random.split(jax.random.PRNGKey(13), 9)
        steps, n = 7, 3
        frames = jax.random.randint(k[0], (steps + 1, n, m.OBS - 1), 0, 256)
        prev = jax.random.randint(k[1], (steps + 1, n, 1), 0, m.A)
        first = 1.0 * (jax.random.uniform(k[2], (steps + 1, n)) < 0.3)
        d = dict(s=jnp.concatenate([frames, prev], -1).astype(jnp.uint8), first=first, done=first[1:],
                 a=jax.random.randint(k[3], (steps, n), 0, m.A), r=jax.random.normal(k[4], (steps, n)),
                 mu=jax.random.normal(k[5], (steps, n, m.A)), h0=jax.random.normal(k[6], (n, m.memory_size())))
        params, eta = m.init_agent(k[7]), m.init_rule(k[8])
        def step(h, x):
            return m.agent_step(params, h, *x, "pixels")
        stepped = jax.lax.scan(step, d["h0"], (d["s"], d["first"]))[1]
        for a, b in zip(jax.tree.leaves(m.unroll(params, d, "pixels")), jax.tree.leaves(stepped)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
        m.CNN_STEPS = 3  # the CNN in passes of 3 time steps
        for a, b in zip(jax.tree.leaves(m.unroll(params, d, "pixels")), jax.tree.leaves(stepped)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
        m.CNN_STEPS = 0
        m.CNN_DTYPE = jnp.float32  # compare the two meta-gradients without bf16 rounding
        zeros = jax.tree.map(jnp.zeros_like, params)
        probe = jax.tree.map(lambda x: jax.random.normal(k[0], x.shape), params)
        def meta(mixflow):
            def f(e):
                new = m.inner_step(params, (zeros, zeros, jnp.zeros(())), e, d, "pixels", mixflow)[0]
                return sum((x * y).sum() for x, y in zip(jax.tree.leaves(new), jax.tree.leaves(probe)))
            with jax.default_matmul_precision("highest"):
                return jax.jit(jax.grad(f))(eta)
        mixed, plain = meta(True), meta(False)
        self.assertGreater(float(m.optax.tree.norm(plain)), 1e-6)
        for a, b in zip(jax.tree.leaves(mixed), jax.tree.leaves(plain)):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=max(1e-6, 3e-3 * float(np.abs(b).max())))
        m.CNN_REMAT = True  # recomputed CNN blocks: the same meta-gradient (up to float noise, e.g. in the pi bias)
        scale = max(float(np.abs(b).max()) for b in jax.tree.leaves(mixed))
        for a, b in zip(jax.tree.leaves(meta(True)), jax.tree.leaves(mixed)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * scale)
    def test_action_slots(self):  # catch through 8 slots: dead slots are illegal, the env plays slots[a]
        m.use_slots(8)
        m.ACT_REDRAW = 0.5
        agent = m.init_agent_state(jax.random.PRNGKey(5))
        slots = np.asarray(agent[4][3])
        self.assertTrue({0, 1, 2} <= set(slots[slots >= 0].tolist()))
        d = m.collect(agent[0], agent[4], jax.random.PRNGKey(6), "catch")[1]
        np.testing.assert_array_equal(d["valid"][0, 0], slots >= 0)
        self.assertTrue((slots[np.asarray(d["a"])] >= 0).all())
        s, kept = np.asarray(d["s"]), np.asarray(d["first"][1:]) == 0
        moved = np.clip(s[:-1, :, 2] + slots[np.asarray(d["a"])] - 1, 0, m.W - 1)
        np.testing.assert_array_equal(s[1:, :, 2][kept], moved[kept])
        loss, (agent, _) = m.meta_loss(self.eta, agent, self.key)  # with redraws
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(agent[4][3].shape, (8,))
        m.RULE_LOGP = True  # the rule also reads log-probabilities; dead slots read the floor
        eta = m.init_rule(jax.random.PRNGKey(8))
        per_action = m.features(m.unroll(agent[0], d, "catch"), d, m.q_stats(m.unroll(agent[0], d, "catch"), d))[1]
        np.testing.assert_array_equal(per_action[..., -2][:, :, slots < 0], -10.0)
        self.assertTrue(np.isfinite(float(m.meta_loss(eta, agent, self.key)[0])))
    def test_rule_ignores_illegal_actions(self):  # RULE_LEGAL: 3 dead slots change nothing on the 5 legal actions
        m.RULE_LEGAL, m.RULE_LOGP = True, True
        k = jax.random.split(jax.random.PRNGKey(11), 8)
        steps, n, legal = 30, 6, jnp.arange(8) < 5
        def logits(key, shape):
            return jnp.where(legal, jax.random.normal(key, shape), -1e9)
        pi, mu = logits(k[0], (steps, n, 8)), logits(k[5], (steps - 1, n, 8))
        out = dict(pi=pi, q=jax.random.normal(k[1], (steps, n, 8)), y=jax.random.normal(k[2], (steps, n, m.D)),
                   z=jax.random.normal(k[3], (steps, n, 8, m.D)))
        roll = dict(a=jax.random.categorical(k[4], pi[:-1]), r=jax.random.normal(k[6], (steps - 1, n)),
                    done=1.0 * (jax.random.uniform(k[7], (steps - 1, n)) < 0.1), mu=mu,
                    valid=jnp.broadcast_to(legal, (steps, n, 8)))
        eta = m.init_rule(jax.random.PRNGKey(12))
        def targets(out, roll):
            m.A = out["pi"].shape[-1]
            return m.rule(eta, *m.features(out, roll, m.q_stats(out, roll)), roll)
        out5 = out | dict(pi=pi[..., :5], q=out["q"][..., :5], z=out["z"][..., :5, :])
        five = targets(out5, roll | dict(mu=mu[..., :5], valid=roll["valid"][..., :5]))
        eight = targets(out, roll)
        for a, b in zip((eight[0][..., :5],) + eight[1:], five):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    def test_anchored_policy_target(self):  # ANCHOR 1: a rule whose pi output is zero keeps the policy; ANCHOR 0.9 with
        m.ANCHOR = 1.0  # the log pi input and its meta-gradient: MixFlow and plain meta-gradients still match
        eta = self.eta | dict(pi=jax.tree.map(jnp.zeros_like, self.eta["pi"]))
        out = m.unroll(self.agent[0], self.rollout, "catch")
        pi = m.anchor(m.rule(eta, *m.features(out, self.rollout, m.q_stats(out, self.rollout)), self.rollout), out)[0]
        np.testing.assert_allclose(m.kl(pi, out["pi"][:-1]), 0.0, atol=1e-6)
        m.ANCHOR, m.RULE_LOGP, m.LOGP_GRAD = 0.9, True, True
        self.eta = m.init_rule(jax.random.PRNGKey(4))
        mixed, plain = self.meta_grad(True), self.meta_grad(False)
        self.assertGreater(float(m.optax.tree.norm(plain["pi"]["w"])), 1e-6)
        scale = max(float(np.abs(b).max()) for b in jax.tree.leaves(plain))  # the pi bias: 0 up to float noise
        for a, b in zip(jax.tree.leaves(mixed), jax.tree.leaves(plain)):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=max(1e-6, 3e-3 * float(np.abs(b).max()), 1e-5 * scale))
    def test_rollout_shapes_and_terminal_flags(self):
        d = self.rollout
        self.assertEqual(d["s"].shape[0], d["a"].shape[0] + 1)
        np.testing.assert_array_equal(d["first"][1:], d["done"])  # a new episode starts after every terminal
    def test_vtrace_on_policy_is_td_lambda(self):
        k = jax.random.split(jax.random.PRNGKey(7), 3)
        v, r = jax.random.normal(k[0], (11, 4)), jax.random.normal(k[1], (10, 4))
        discount = 0.9 * (jax.random.uniform(k[2], (10, 4)) > 0.2)
        targets, advantages = m.vtrace(v, r, discount, jnp.ones((10, 4)), 0.8)
        np.testing.assert_allclose(targets, m.lambda_returns(r, discount, v[1:], 0.8), rtol=1e-5, atol=1e-5)
        next_targets = jnp.concatenate([targets[1:], v[-1:]])
        np.testing.assert_allclose(advantages, r + discount * next_targets - v[:-1], rtol=1e-5, atol=1e-5)
    def test_retrace_cuts_traces(self):
        k = jax.random.split(jax.random.PRNGKey(8), 4)
        r, v, q = (jax.random.normal(x, (6, 2)) for x in k[:3])
        discount = jnp.full((6, 2), 0.9)
        one_step = m.retrace(r, discount, v, q, jnp.zeros((6, 2)))  # c = 0: one-step expected-SARSA targets
        np.testing.assert_allclose(one_step, r + discount * v, rtol=1e-6, atol=1e-6)
        full = m.retrace(r, discount, v, q, jnp.ones((6, 2)))  # c = 1: G_t = r_t + d (v' + G' - q')
        np.testing.assert_allclose(full[-2], r[-2] + 0.9 * (v[-2] + full[-1] - q[-2]), rtol=1e-6, atol=1e-6)
    def test_replay_buffer(self):
        buffer = m.init_buffer()
        n = len(buffer["a"])
        for _ in range(n // m.B + 1):
            buffer = m.store(buffer, self.rollout)
        self.assertEqual(int(buffer["size"]), n)  # full, and the write position wrapped around
        self.assertEqual(int(buffer["idx"]), (n // m.B + 1) * m.B % n)
        batch = m.sample(buffer, self.rollout, self.key)
        self.assertEqual(batch["s"].shape, (m.ROLL + 1, m.BATCH, 3))
        self.assertEqual(batch["h0"].shape, (m.BATCH, m.HID))
    def test_updates_per_rollout(self):  # UTD 1 = the old update; UTD 2: two finite Adam steps; meta-gradient flows
        one = m.agent_update(self.eta, self.agent, self.key)[0]  # BATCH = B: learning starts at once
        key_collect, key_sample = jax.random.split(self.key)
        rollout = m.collect(self.agent[0], self.agent[4], key_collect, "catch")[1]
        batch = m.sample(m.store(self.agent[5], rollout), rollout, key_sample)
        old = m.inner_step(self.agent[0], self.agent[1], self.eta, batch, "catch")[0]
        for x, y in zip(jax.tree.leaves(one[0]), jax.tree.leaves(old)):
            np.testing.assert_allclose(x, y, atol=1e-6)
        m.UTD = 2
        two = m.agent_update(self.eta, self.agent, self.key)[0]
        self.assertEqual((int(one[1][2]), int(two[1][2])), (1, 2))  # Adam step counts
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(two[0])))
        self.assertGreater(float(m.optax.tree.norm(jax.tree.map(lambda x, y: x - y, two[0], one[0]))), 1e-4)
        np.testing.assert_array_equal(one[5]["a"], two[5]["a"])  # the same rollout went into the buffer
        g = jax.grad(lambda e: m.meta_loss(e, self.agent, self.key)[0])(self.eta)
        self.assertGreater(float(m.optax.tree.norm(g)), 1e-6)
    def test_rules_of_one_run_share_flags(self):  # a mix would run every rule with the last one's flags: refused
        saved = m.obs, m.B, m.ROLL, m.RETRACE
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        import bench  # sets mini's obs, B, ROLL, RETRACE for its scripts: put them back
        m.obs, m.B, m.ROLL, m.RETRACE = saved
        with tempfile.TemporaryDirectory() as d:
            paths = [f"{d}/{resid}.npz" for resid in (False, True)]
            for path, resid in zip(paths, (False, True)):
                eta = m.init_rule(self.key)
                weights = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_leaves_with_path(eta)}
                np.savez(path, **weights, **{"flag RULE_LEGAL": False, "flag RULE_RESID": resid})
            self.assertEqual(len(bench.load_rules(paths[:1] * 2)), 2)
            with self.assertRaises(ValueError):
                bench.load_rules(paths)
    def test_no_q_variant_has_no_value_path(self):
        m.USE_Q = False
        try:
            eta = m.init_rule(jax.random.PRNGKey(5))
            params = m.inner_step(self.agent[0], self.agent[1], eta, self.rollout, "catch")[0]
            np.testing.assert_array_equal(params["q"]["w"], self.agent[0]["q"]["w"])  # Q head never trained
            g = jax.grad(lambda e: m.meta_loss(e, self.agent, self.key)[0])(eta)
            self.assertGreater(float(m.optax.tree.norm(g)), 1e-6)
            frozen = m.run_frozen(eta, jax.random.PRNGKey(6), 2)
            self.assertTrue(np.isfinite(frozen).all())
        finally:
            m.USE_Q = True
    def test_random_lifetimes(self):
        lifetimes = np.array(m.LIFETIMES)
        drawn = np.asarray(m.draw_lifetimes(jax.random.PRNGKey(4), m.LIFETIMES, 200_000))
        experience = np.array([drawn[drawn == x].sum() for x in lifetimes]) / drawn.sum()
        np.testing.assert_allclose(experience, 1 / len(lifetimes), atol=0.01)  # same experience in every category
        eta, _, logs = jax.jit(lambda k: m.train(k, 6, lifetimes=(2, 3)))(jax.random.PRNGKey(0))
        self.assertTrue(all(np.isfinite(v).all() for v in logs.values()))
        self.assertLess(float(logs["mean_age"].max()), 3)  # every agent restarted by its third meta-step
    def test_one_jit_training_and_frozen_control(self):
        run = jax.jit(lambda k, s: m.train(k, 3, s))
        eta, _, logs = run(jax.random.PRNGKey(0), 1.0)
        self.assertTrue(all(np.isfinite(v).all() for v in logs.values()))
        frozen, _, _ = run(jax.random.PRNGKey(0), 0.0)
        eta0 = m.init_rule(jax.random.split(jax.random.PRNGKey(0), 3)[0])
        for b, c in zip(jax.tree.leaves(frozen), jax.tree.leaves(eta0)):
            np.testing.assert_allclose(b, c, atol=1e-6)  # init inside vs outside jit: float roundoff only
        self.assertGreater(float(m.optax.tree.norm(jax.tree.map(lambda a, c: a - c, eta, eta0))), 1e-4)

if __name__ == "__main__":
    unittest.main()
