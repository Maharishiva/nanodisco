"""Checks for scripts/minatar.py; run: python -m unittest test_minatar -v. CPU, about a minute."""

import functools
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import minatar as mn

N, L, U, R, D, F = range(6)
# Random policy (uniform over the 6 actions, no sticky actions, episodes cut at 1000 steps, Freeway at its own
# 2501): mean return and its standard error in the original MinAtar 1.0.15 code, 4000 episodes (Freeway 1200).
RANDOM = dict(asterix=(0.498, 0.014), freeway=(0.128, 0.010), seaquest=(0.093, 0.006), space_invaders=(2.986, 0.038))

def one(game, key=0, **fields):  # a single env of one game, some fields replaced
    s = mn.RESETS[mn.GAMES.index(game)](jax.random.PRNGKey(key))
    return s | {k: jnp.asarray(v, s[k].dtype) for k, v in fields.items()}

def act(game, s, a, key=0):
    return mn.STEPS[mn.GAMES.index(game)](jax.random.PRNGKey(key), s, a)

def rows(*objects, n):  # Seaquest objects from (x, y, right, move timer, shot timer) tuples
    a = np.zeros((n, 6), np.int32)
    for i, (x, y, right, mt, st) in enumerate(objects):
        a[i] = [x, y, right, 1, mt, st]
    return {c: jnp.asarray(a[:, i]) for i, c in enumerate(mn.COLS)}

def alive(o):  # live Seaquest objects as rows
    a = np.stack([np.asarray(o[c]) for c in mn.COLS], 1)
    return a[a[:, 3] == 1].tolist()

@functools.partial(jax.jit, static_argnums=2)
def play(key, s, steps):  # uniform random actions over the full set
    def body(s, k):
        ka, ks = jax.random.split(k)
        s, r, done = mn.step(ks, s, jax.random.randint(ka, s["time"].shape, 0, mn.A))
        return s, (mn.obs(s), r, done)
    return jax.lax.scan(body, s, jax.random.split(key, steps))

class Interface(unittest.TestCase):
    def test_shapes(self):
        for g in range(len(mn.GAMES)):
            s = mn.reset(jax.random.PRNGKey(0), g, 5)
            o, v = mn.obs(s), mn.valid(s)
            self.assertEqual((o.shape, o.dtype, v.shape, v.dtype), ((5, mn.OBS), jnp.float32, (5, mn.A), jnp.bool_))
            s, (o, r, d) = play(jax.random.PRNGKey(1), s, 50)
            self.assertEqual((o.shape, r.shape, r.dtype, d.dtype), ((50, 5, mn.OBS), (50, 5), jnp.float32, jnp.bool_))
            channels = mn.OBSERVE[g](jax.tree.map(lambda x: x[0], s[mn.GAMES[g]])).shape[-1]
            self.assertEqual(float(o.reshape(50, 5, 10, 10, mn.C)[..., channels:].sum()), 0.0)  # zero padding
            self.assertTrue(set(np.unique(o)) <= {0.0, 1.0})
        self.assertEqual(mn.valid(mn.reset(jax.random.PRNGKey(0), "freeway", 2)).tolist(), [[1, 0, 1, 0, 1, 0]] * 2)

    def test_determinism(self):
        for g in mn.GAMES:
            s = mn.reset(jax.random.PRNGKey(0), g, 8)
            a, b, c = (play(jax.random.PRNGKey(k), s, 300)[1] for k in (1, 1, 2))
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
            if g != "space_invaders":  # the only game without randomness
                self.assertFalse(np.array_equal(a[0], c[0]))

    def test_auto_reset(self):
        for g in (0, 2, 3):  # games whose reset is deterministic
            s, (o, r, d) = play(jax.random.PRNGKey(0), mn.reset(jax.random.PRNGKey(0), g, 16), 200)
            self.assertGreater(int(d.sum()), 0)
            first = mn.obs(mn.reset(jax.random.PRNGKey(5), g, 1))[0]
            for t, e in zip(*np.nonzero(np.asarray(d))):  # the observation after a done is a fresh game's
                np.testing.assert_array_equal(o[t, e], first)
        s = mn.reset(jax.random.PRNGKey(0), "asterix", 2)
        ent = s["asterix"]["ent"].at[0, 4].set(jnp.array([5, 0, 0, 1]))  # an enemy on the player
        s = s | dict(asterix=s["asterix"] | dict(ent=ent), time=jnp.array([3, mn.LIMIT[0] - 1]))
        s, r, done = mn.step(jax.random.PRNGKey(1), s, jnp.array([N, N]))
        self.assertEqual(done.tolist(), [True, True])  # caught by the enemy; the time limit
        self.assertEqual(s["time"].tolist(), [0, 0])
        for k, v in one("asterix").items():
            np.testing.assert_array_equal(s["asterix"][k][1], v)

    def test_freeway_time_limit(self):  # MinAtar's 2500-frame timer ends Freeway on its 2501st step
        s = mn.reset(jax.random.PRNGKey(0), "freeway", 1)
        s = s | dict(time=jnp.array([2499]))
        s, _, done = mn.step(jax.random.PRNGKey(1), s, jnp.array([N]))
        self.assertFalse(bool(done[0]))
        s, _, done = mn.step(jax.random.PRNGKey(1), s, jnp.array([N]))
        self.assertTrue(bool(done[0]))

    def test_invalid_actions_do_nothing(self):
        for g in range(len(mn.GAMES)):
            s = jax.tree.map(lambda x: x[0], play(jax.random.PRNGKey(3), mn.reset(jax.random.PRNGKey(0), g, 1), 40)[0])
            base = mn.STEPS[g](jax.random.PRNGKey(4), s[mn.GAMES[g]], N)
            for a in np.flatnonzero(~np.asarray(mn.VALID[g])):
                out = mn.STEPS[g](jax.random.PRNGKey(4), s[mn.GAMES[g]], a)
                jax.tree.map(np.testing.assert_array_equal, out, base)

    def test_vmap_over_games(self):  # a batched game index: the switch runs as a select, same results
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        s = jax.vmap(lambda k, g: mn.reset(k, g, 3))(keys, jnp.arange(4))
        a = jnp.array([[1, 2, 3], [2, 4, 0], [5, 3, 4], [5, 1, 3]])
        out = jax.vmap(mn.step)(keys, s, a)
        o = jax.vmap(mn.obs)(out[0])
        for g in range(4):
            si = jax.tree.map(lambda x: x[g], s)
            ref = mn.step(keys[g], si, a[g], g)
            jax.tree.map(np.testing.assert_array_equal, jax.tree.map(lambda x: x[g], out), ref)
            np.testing.assert_array_equal(o[g], mn.obs(ref[0], g))

    def test_random_returns(self):  # finished-episode returns of a uniform random policy vs the original MinAtar
        for g, (n, steps) in dict(asterix=(128, 2000), freeway=(128, 2600), seaquest=(256, 500),
                                  space_invaders=(128, 1000)).items():
            _, (_, r, d) = play(jax.random.PRNGKey(7), mn.reset(jax.random.PRNGKey(6), g, n), steps)
            r, d = np.asarray(r), np.asarray(d)
            rets = []
            for e in range(n):  # the last, unfinished episode of each env is dropped
                rets += [x.sum() for x in np.split(r[:, e], np.flatnonzero(d[:, e]) + 1)[:-1]]
            mean, se = np.mean(rets), np.std(rets) / np.sqrt(len(rets))
            ref, ref_se = RANDOM[g]
            with self.subTest(game=g, mean=mean, ref=ref):
                self.assertLess(abs(mean - ref), 0.05 * ref + 4 * np.hypot(se, ref_se))

class Asterix(unittest.TestCase):
    def ent(self, *rows):  # (slot, x, moves right, gold)
        e = np.zeros((8, 4), np.int32)
        for i, x, right, gold in rows:
            e[i] = [x, right, gold, 1]
        return e

    def test_gold_and_enemy(self):
        s = one("asterix", ent=self.ent((4, 6, 0, 1)), move_timer=0, spawn_timer=3)  # gold in row 5 moves onto (5, 5)
        s, r, term = act("asterix", s, N)
        self.assertEqual((float(r), bool(term), int(s["ent"][4, 3])), (1.0, False, 0))
        s = one("asterix", ent=self.ent((4, 6, 0, 0)), move_timer=0, spawn_timer=3)
        self.assertTrue(bool(act("asterix", s, N)[2]))  # an enemy does the same: game over
        s = one("asterix", ent=self.ent((4, 6, 0, 1)), move_timer=2, spawn_timer=3)
        s, r, _ = act("asterix", s, R)  # the player steps onto the gold
        self.assertEqual((int(s["x"]), float(r)), (6, 1.0))

    def test_moves_and_spawn(self):
        s, _, _ = act("asterix", one("asterix", y=1, spawn_timer=3), U)
        self.assertEqual(int(s["y"]), 1)  # row 0 is out of bounds
        s, _, _ = act("asterix", one("asterix", y=8, x=9, spawn_timer=3), D)
        self.assertEqual((int(s["y"]), int(s["x"])), (8, 9))
        for k in range(5):
            s, _, _ = act("asterix", one("asterix", spawn_timer=0), N, key=k)
            new = [row for row in np.asarray(s["ent"]).tolist() if row[3]]
            self.assertEqual(len(new), 1)
            x, right, _, _ = new[0]
            self.assertEqual((x, int(s["spawn_timer"])), (0 if right else 9, 9))

    def test_ramp(self):
        s, _, _ = act("asterix", one("asterix", ramp_timer=-1, spawn_timer=3), N)
        self.assertEqual([int(s[k]) for k in ("spawn_speed", "move_speed", "ramp_index", "ramp_timer")], [9, 5, 1, 100])
        s, _, _ = act("asterix", one("asterix", ramp_timer=-1, ramp_index=1, spawn_timer=3), N)
        self.assertEqual([int(s[k]) for k in ("spawn_speed", "move_speed", "ramp_index")], [9, 4, 2])

class Freeway(unittest.TestCase):
    def test_cross(self):
        s = one("freeway", pos=1, move_timer=0)
        s, r, _ = act("freeway", s, U, key=3)
        self.assertEqual((float(r), int(s["pos"]), int(s["move_timer"])), (1.0, 9, 2))
        np.testing.assert_array_equal(s["car_t"], np.abs(s["car_v"]) - 1)  # new speeds, timers restarted
        s, r, _ = act("freeway", one("freeway", pos=5, move_timer=2), U)
        self.assertEqual((float(r), int(s["pos"]), int(s["move_timer"])), (0.0, 5, 1))  # cooling down

    def test_cars(self):
        v = np.array([2, 1, 1, 1, 1, 1, 1, 1])
        s = one("freeway", pos=3, car_x=np.array([9, 0, 4, 0, 0, 0, 0, 0]), car_t=np.array([0, 3, 2, 3, 3, 3, 3, 3]),
                car_v=v)
        s, _, _ = act("freeway", s, N)
        self.assertEqual(int(s["pos"]), 9)  # car 2 sits on the chicken's cell in row 3
        self.assertEqual((int(s["car_x"][0]), int(s["car_t"][0])), (0, 2))  # wraps around, timer restarts
        s = one("freeway", pos=3, car_x=np.array([0, 0, 3, 0, 0, 0, 0, 0]), car_t=np.array([3, 3, 0, 3, 3, 3, 3, 3]),
                car_v=v)
        s, _, _ = act("freeway", s, N)
        self.assertEqual(int(s["pos"]), 9)  # car 2 moves onto the chicken
        o = mn.freeway_obs(one("freeway", car_x=np.zeros(8, int), car_v=np.array([-3, 1, 1, 1, 1, 1, 1, 1])))
        self.assertEqual((o[1, 1, 4], o[2, 9, 2], o[9, 4, 0]), (1, 1, 1))  # trail channel by speed, wrapped

class Seaquest(unittest.TestCase):
    def sea(self, **kw):
        kw = dict(e_timer=5, d_timer=5) | kw
        objects = {k: rows(*kw.pop(k, ()), n=n) for k, n in mn.SLOTS.items()}
        return one("seaquest", **kw) | objects

    def test_shoot_fish(self):
        s = self.sea(x=5, y=3, right=1, fish=[(6, 3, 0, 3, 0)])
        s, r, term = act("seaquest", s, F)
        got = (float(r), bool(term), alive(s["fish"]), alive(s["fb"]), int(s["shot"]))
        self.assertEqual(got, (1.0, False, [], [], 4))
        s, r, _ = act("seaquest", self.sea(x=5, y=3, right=1, shot=2, fish=[(6, 3, 0, 3, 0)]), F)
        self.assertEqual(float(r), 0.0)  # cooling down: no bullet

    def test_surface(self):
        _, _, term = act("seaquest", self.sea(y=1, surface=False), U)
        self.assertTrue(bool(term))  # surfacing without a diver
        s, r, term = act("seaquest", self.sea(y=1, surface=False, held=6, oxygen=150), U)
        self.assertEqual((float(r), bool(term), int(s["held"]), int(s["oxygen"])), (7.0, False, -1, 200))
        self.assertEqual((int(s["e_speed"]), int(s["ramp_index"])), (19, 1))
        s, r, _ = act("seaquest", self.sea(y=1, surface=False, held=2, oxygen=150), U)
        self.assertEqual((float(r), int(s["held"]), int(s["oxygen"])), (0.0, 1, 200))
        _, _, term = act("seaquest", self.sea(y=3, oxygen=0), N)
        self.assertTrue(bool(term))  # out of oxygen
        s, _, _ = act("seaquest", self.sea(y=3, oxygen=100), N)
        self.assertEqual(int(s["oxygen"]), 99)

    def test_divers_and_enemies(self):
        s, _, _ = act("seaquest", self.sea(x=5, y=2, divers=[(5, 3, 1, 2, 0)]), D)
        self.assertEqual((int(s["held"]), alive(s["divers"])), (1, []))
        s, _, term = act("seaquest", self.sea(x=5, y=3, subs=[(2, 5, 1, 3, 0)]), N)
        self.assertEqual((bool(term), alive(s["eb"])), (False, [[3, 5, 1, 1, 0, 0]]))  # fired, then moved
        _, _, term = act("seaquest", self.sea(x=5, y=3, fish=[(6, 3, 0, 0, 0)]), N)
        self.assertTrue(bool(term))  # a fish swims into the sub
        both = [(x, y, right, 9, 0) for y in range(1, 9) for x, right in ((1, 1), (8, 0))]
        s, _, _ = act("seaquest", self.sea(x=5, y=0, e_timer=0, fish=both), N)
        self.assertEqual((len(alive(s["fish"])), alive(s["subs"])), (16, []))  # no row is free for a spawn
        for k in range(5):  # a spawn: x 0 moving right or x 9 moving left, rows 1..8, timer = move speed, then ticks
            s, _, _ = act("seaquest", self.sea(e_timer=0), N, key=k)
            new = alive(s["fish"]) + alive(s["subs"])
            self.assertEqual(len(new), 1)
            x, y, right, _, mt, _ = new[0]
            self.assertEqual((x, mt, 1 <= y <= 8), (0 if right else 9, 4, True))

class SpaceInvaders(unittest.TestCase):
    def inv(self, aliens=None, **kw):
        a = np.zeros((10, 10), bool)
        for y, x in aliens or ():
            a[y, x] = True
        return one("space_invaders", **kw) | ({} if aliens is None else dict(aliens=jnp.asarray(a)))

    def test_shoot_and_new_wave(self):
        f = np.zeros((10, 10), bool)
        f[8, 5] = True
        s = self.inv([(7, 5)], f=f, move_timer=5, alien_shot=5)
        s, r, term = act("space_invaders", s, N)
        self.assertEqual((float(r), bool(term), int(s["interval"]), int(s["ramp_index"])), (1.0, False, 11, 1))
        np.testing.assert_array_equal(s["aliens"], one("space_invaders")["aliens"])  # a new, faster wave
        s, _, _ = act("space_invaders", one("space_invaders"), F)
        self.assertEqual((bool(s["f"][8, 5]), int(s["shot"])), (True, 4))
        s, _, _ = act("space_invaders", one("space_invaders", shot=3), F)
        self.assertEqual((int(s["f"].sum()), int(s["shot"])), (0, 2))

    def test_aliens(self):
        s, _, _ = act("space_invaders", self.inv([(2, 0), (3, 1)], move_timer=0, alien_shot=5), N)
        self.assertEqual((sorted(zip(*np.nonzero(np.asarray(s["aliens"])))), int(s["dir"])), ([(3, 0), (4, 1)], 1))
        self.assertEqual(int(s["move_timer"]), 1)  # min(aliens, interval) - 1
        s, _, _ = act("space_invaders", self.inv([(2, 4), (3, 5)], move_timer=0, alien_shot=5), N)
        self.assertEqual(sorted(zip(*np.nonzero(np.asarray(s["aliens"])))), [(2, 3), (3, 4)])
        s, _, _ = act("space_invaders", one("space_invaders", pos=8, alien_shot=0), N)
        self.assertEqual(np.argwhere(np.asarray(s["e"])).tolist(), [[3, 7]])  # lowest alien of the nearest column
        e = np.zeros((10, 10), bool)
        e[8, 5] = True
        _, _, term = act("space_invaders", one("space_invaders", e=e), N)
        self.assertTrue(bool(term))  # an enemy bullet reaches the cannon

if __name__ == "__main__":
    unittest.main()
