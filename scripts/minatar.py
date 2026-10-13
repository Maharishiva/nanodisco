"""MinAtar Asterix, Freeway, Seaquest and SpaceInvaders in pure JAX, an env module for mini.use_env and bench.

Ported from the original MinAtar (Young & Tian 2019, kenjyoung/MinAtar 1.0.15), with gymnax 1.0.0 as JAX reference.
Conventions as scripts/breakout.py: no sticky actions, MinAtar's difficulty ramping, and an episode ends on
termination or after 1000 steps (Freeway: its own 2500-frame timer, 2501 steps). Actions are MinAtar's full set
n l u r d f; as in MinAtar, an action outside a game's minimal set (valid) does nothing.
Observation: the game's 10x10xC channels zero-padded to C = 10 (Seaquest's), flattened.
State: dict with per-env game index and step count, and one sub-dict per game (the other games' stay at reset).
All envs of one reset share a game, so step and obs lax.switch on the first env's game; under vmap over games the
switch becomes a select and runs all four; a static game index passed to step and obs runs only that game.
Seaquest keeps objects in fixed slots (SLOTS): a spawn into a full array is dropped, and when two objects of one
kind share a cell, which one a bullet or the sub takes may differ from MinAtar's list order.
"""

import jax
import jax.numpy as jnp

OBS, A, C = 1000, 6, 10
GAMES = ("asterix", "freeway", "seaquest", "space_invaders")
N, L, U, R, D, F = range(6)
NOOP = N  # legal in every game; mini.draw_slots pads extra slots with it (without it, ACT_PAD noop/mix copy)
LIMIT = (1000, 2501, 1000, 1000)
VALID = jnp.array([[1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 0, 1]], bool)
SLOTS = dict(fb=4, eb=16, fish=24, subs=16, divers=8)  # Seaquest: friendly and enemy bullets, fish, subs, divers
COLS = ("x", "y", "dir", "on", "mt", "st")  # Seaquest object columns: position, direction (1 right), alive, timers
CELLS = jnp.arange(10)
ROWS = jnp.arange(1, 9)  # Asterix slots and Freeway cars live in rows 1..8

def i32(x):
    return jnp.asarray(x, jnp.int32)

def down(t):  # a timer that stops at 0
    return t - (t > 0)

def lr(x, a):  # horizontal move
    return jnp.where(a == L, jnp.maximum(0, x - 1), jnp.where(a == R, jnp.minimum(9, x + 1), x))

def ud(y, a, top, bottom):  # vertical move
    return jnp.where(a == U, jnp.maximum(top, y - 1), jnp.where(a == D, jnp.minimum(bottom, y + 1), y))

# ---------------------------------------------------------------- Asterix
def asterix_reset(key):  # ent rows (slot i in row i+1): x, moves right, gold, alive
    return dict(x=i32(5), y=i32(5), ent=jnp.zeros((8, 4), jnp.int32), spawn_speed=i32(10), spawn_timer=i32(10),
                move_speed=i32(5), move_timer=i32(5), ramp_timer=i32(100), ramp_index=i32(0))

def asterix_step(key, s, a):
    k1, k2, k3 = jax.random.split(key, 3)
    e, right, gold = s["ent"], jax.random.uniform(k1) < 0.5, jax.random.uniform(k2) < 1 / 3
    free = e[:, 3] == 0
    slot = jax.random.categorical(k3, jnp.where(free, 0.0, -jnp.inf))  # a random free slot
    new = jnp.stack([jnp.where(right, 0, 9), right, gold, True]).astype(jnp.int32)
    spawn = s["spawn_timer"] == 0
    e = jnp.where(spawn & free.any(), e.at[slot].set(new), e)
    x, y = lr(s["x"], a), ud(s["y"], a, 1, 8)
    def touch(e):  # gold under the player is taken (+1), an enemy ends the game
        hit = (e[:, 3] == 1) & (e[:, 0] == x) & (ROWS == y)
        got = hit & (e[:, 2] == 1)
        return e.at[:, 3].set(jnp.where(got, 0, e[:, 3])), got.sum(), (hit & (e[:, 2] == 0)).any()
    e, r1, t1 = touch(e)
    move = s["move_timer"] == 0
    ex = e[:, 0] + move * e[:, 3] * (2 * e[:, 1] - 1)
    e = e.at[:, 0].set(ex).at[:, 3].set(jnp.where((ex < 0) | (ex > 9), 0, e[:, 3]))
    e, r2, t2 = touch(e)  # without a move this finds nothing new
    e = e * (e[:, 3:] == 1)  # free slots are zero rows
    ramp = (s["spawn_speed"] > 1) | (s["move_speed"] > 1)
    up = ramp & (s["ramp_timer"] < 0)
    return dict(
        x=x,
        y=y,
        ent=e,
        spawn_speed=s["spawn_speed"] - (up & (s["spawn_speed"] > 1)),
        spawn_timer=jnp.where(spawn, s["spawn_speed"], s["spawn_timer"]) - 1,
        move_speed=s["move_speed"] - (up & (s["move_speed"] > 1) & (s["ramp_index"] % 2 == 1)),
        move_timer=jnp.where(move, s["move_speed"], s["move_timer"]) - 1,
        ramp_timer=jnp.where(up, 100, s["ramp_timer"] - ramp),
        ramp_index=s["ramp_index"] + up,
    ), r1 + r2, t1 | t2

def asterix_obs(s):  # player, enemy, trail, gold
    e, on = s["ent"], 1.0 * (s["ent"][:, 3] == 1)
    o = jnp.zeros((10, 10, 4)).at[s["y"], s["x"], 0].set(1).at[ROWS, jnp.clip(e[:, 0], 0, 9), 1 + 2 * e[:, 2]].max(on)
    back = e[:, 0] + 1 - 2 * e[:, 1]
    return o.at[ROWS, jnp.clip(back, 0, 9), 2].max(on * ((back >= 0) & (back <= 9)))

# ---------------------------------------------------------------- Freeway
def freeway_speeds(key):  # signed car speeds 1..5 (a car moves every |v| + 1 frames), random directions
    k1, k2 = jax.random.split(key)
    return jax.random.randint(k1, (8,), 1, 6) * jnp.where(jax.random.uniform(k2, (8,)) < 0.5, -1, 1)

def freeway_reset(key):  # car i in row i+1
    v = freeway_speeds(key)
    return dict(pos=i32(9), move_timer=i32(3), car_x=jnp.zeros(8, jnp.int32), car_t=jnp.abs(v), car_v=v)

def freeway_step(key, s, a):
    go = s["move_timer"] == 0
    pos = jnp.where(go, ud(s["pos"], a, 0, 9), s["pos"])
    move_timer = jnp.where(go & ((a == U) | (a == D)), 3, s["move_timer"])
    win = pos == 0  # +1, new car speeds, back to the bottom
    v = jnp.where(win, freeway_speeds(key), s["car_v"])
    t = jnp.where(win, jnp.abs(v), s["car_t"])
    pos = jnp.where(win, 9, pos)
    move = t == 0
    x = jnp.where(move, (s["car_x"] + jnp.sign(v)) % 10, s["car_x"])
    hit = (ROWS == pos) & ((s["car_x"] == 4) | (move & (x == 4)))  # a car hits the chicken: back to the bottom
    return dict(pos=jnp.where(hit.any(), 9, pos), move_timer=down(move_timer), car_x=x,
                car_t=jnp.where(move, jnp.abs(v), t - 1), car_v=v), 1.0 * win, jnp.array(False)

def freeway_obs(s):  # chicken, car, trail channel by speed
    o = jnp.zeros((10, 10, 7)).at[s["pos"], 4, 0].set(1).at[ROWS, s["car_x"], 1].set(1)
    return o.at[ROWS, (s["car_x"] - jnp.sign(s["car_v"])) % 10, 1 + jnp.abs(s["car_v"])].set(1)

# ---------------------------------------------------------------- Seaquest
def seaquest_reset(key):  # objects: dict of columns, each [slots]
    return {k: {c: jnp.zeros(n, jnp.int32) for c in COLS} for k, n in SLOTS.items()} | dict(
        oxygen=i32(200), held=i32(0), x=i32(5), y=i32(0), right=i32(0), shot=i32(0), surface=jnp.array(True),
        e_speed=i32(20), e_timer=i32(20), d_timer=i32(30), move_speed=i32(5), ramp_index=i32(0))

def before(m):  # how many True precede each entry along the last axis
    k = m.shape[-1]
    return (m[..., None, :] & jnp.tri(k, k, -1, bool)).sum(-1)

def first(m):  # the first True along the last axis
    return m & (before(m) == 0)

def add(o, row, ok):  # row into the first free slot; dropped if none
    put = first(o["on"] == 0) & ok
    return {c: jnp.where(put, row[i], o[c]) for i, c in enumerate(COLS)}

def add_many(o, rows, ok):  # rows where ok, in order, into the free slots, in order
    free = o["on"] == 0
    m = free[:, None] & ok & (before(free)[:, None] == before(ok))  # [slot, row]
    return {c: jnp.where(m.any(1), (m * rows[c]).sum(1), o[c]) for c in COLS}

def at(o, x, y):
    return (o["on"] == 1) & (o["x"] == x) & (o["y"] == y)

def shift(o, go):  # objects where go move one cell along their direction; leaving the board removes them
    x = o["x"] + go * (2 * o["dir"] - 1)
    return o | dict(x=x, on=o["on"] * ((x >= 0) & (x <= 9)))

def tick(o, speed):  # objects whose move timer is 0 move and restart it at speed; the others count down
    go = (o["on"] == 1) & (o["mt"] == 0)
    return shift(o | dict(mt=jnp.where(go, speed, jnp.maximum(o["mt"] - 1, 0))), go), go

def collide(b, t, ok):  # each bullet removes the first target (where ok) on its cell, one bullet per target
    m = first((b["on"][:, None] == 1) & at(t, b["x"][:, None], b["y"][:, None]) & ok)
    m = first(m.T).T
    return b | dict(on=b["on"] * ~m.any(1)), t | dict(on=t["on"] * ~m.any(0)), m.sum()

def seaquest_step(key, s, a):
    k = jax.random.split(key, 5)
    right, sub = jax.random.uniform(k[0]) < 0.5, jax.random.uniform(k[1]) < 1 / 3
    y = jax.random.randint(k[2], (), 1, 9)
    def clash(o):  # an enemy in that row moving the other way blocks the spawn
        return ((o["on"] == 1) & (o["y"] == y) & (o["dir"] != right)).any()
    spawn = (s["e_timer"] == 0) & ~clash(s["fish"]) & ~clash(s["subs"])
    new = (jnp.where(right, 0, 9), y, right, 1, s["move_speed"])
    fish, subs = add(s["fish"], new + (0,), spawn & ~sub), add(s["subs"], new + (10,), spawn & sub)
    d_right = jax.random.uniform(k[3]) < 0.5
    divers = add(s["divers"], (jnp.where(d_right, 0, 9), jax.random.randint(k[4], (), 1, 9), d_right, 1, 5, 0),
                 s["d_timer"] == 0)
    fire = (a == F) & (s["shot"] == 0)
    fb = add(s["fb"], (s["x"], s["y"], s["right"], 1, 0, 0), fire)
    x, y = lr(s["x"], a), ud(s["y"], a, 0, 8)
    fb = shift(fb, fb["on"] == 1)  # friendly bullets: fish first, then subs
    fb, fish, r1 = collide(fb, fish, True)
    fb, subs, r2 = collide(fb, subs, True)
    pre = at(divers, x, y)  # divers: picked up on reaching the sub, at most 6 held
    divers, go = tick(divers, 5)
    take = pre | (go & at(divers, x, y))
    take = take & (before(take) < 6 - s["held"])
    divers, held = divers | dict(on=divers["on"] * ~take), s["held"] + take.sum()
    term, live = at(subs, x, y).any(), subs["on"] == 1  # enemy subs move, may be shot, and fire
    subs, go = tick(subs, s["move_speed"])
    hit = go & at(subs, x, y)
    fb, subs, r3 = collide(fb, subs, go & ~hit)
    fires = live & (subs["st"] == 0)  # subs removed this step still fire, as in MinAtar
    shots = subs | dict(on=jnp.ones_like(subs["on"]), mt=0 * subs["mt"], st=0 * subs["st"])
    eb = add_many(s["eb"], shots, fires & (subs["x"] >= 0) & (subs["x"] <= 9))
    subs = subs | dict(st=jnp.where(fires, 10, subs["st"] - 1))
    term = term | hit.any() | at(eb, x, y).any()  # enemy bullets
    eb = shift(eb, eb["on"] == 1)
    term = term | at(eb, x, y).any() | at(fish, x, y).any()  # enemy fish
    fish, go = tick(fish, s["move_speed"])
    hit = go & at(fish, x, y)
    fb, fish, r4 = collide(fb, fish, go & ~hit)
    under = y > 0  # oxygen, and surfacing: needs a diver; 6 divers pay the remaining oxygen
    surfacing = ~under & ~s["surface"]
    up = surfacing & (held != 0)
    full = held == 6
    ramp = up & ((s["e_speed"] > 1) | (s["move_speed"] > 2))
    reward = r1 + r2 + r3 + r4 + jnp.where(up & full, s["oxygen"] * 10 // 200, 0)
    term = term | hit.any() | (s["oxygen"] <= 0) | (surfacing & (held == 0))
    return dict(
        fb=fb,
        eb=eb,
        fish=fish,
        subs=subs,
        divers=divers,
        oxygen=jnp.where(under, s["oxygen"] - 1, jnp.where(up, 200, s["oxygen"])),
        held=jnp.where(up, jnp.where(full, -1, held - 1), held),  # MinAtar's count after 6 divers is -1
        x=x,
        y=y,
        right=jnp.where(a == L, 0, jnp.where(a == R, 1, s["right"])),
        shot=down(jnp.where(fire, 5, s["shot"])),
        surface=~under & (s["surface"] | up),
        e_speed=s["e_speed"] - (ramp & (s["e_speed"] > 1)),
        e_timer=down(jnp.where(s["e_timer"] == 0, s["e_speed"], s["e_timer"])),
        d_timer=down(jnp.where(s["d_timer"] == 0, 30, s["d_timer"])),
        move_speed=s["move_speed"] - (ramp & (s["move_speed"] > 2) & (s["ramp_index"] % 2 == 1)),
        ramp_index=s["ramp_index"] + ramp,
    ), reward, term

def grid(on, y, x):  # [10, 10]: the cells of the objects where on
    return (on[:, None, None] & (y[:, None, None] == CELLS[:, None]) & (x[:, None, None] == CELLS)).any(0)

def back(o):  # objects' previous cells, the trail
    x = o["x"] + 1 - 2 * o["dir"]
    return grid((o["on"] == 1) & (x >= 0) & (x <= 9), o["y"], x)

def seaquest_obs(s):  # sub front, back, friendly bullet, trail, enemy bullet, fish, sub, oxygen, divers held, diver
    def cell(y, x):
        return grid(jnp.ones(1, bool), y[None], x[None])
    def objects(k):
        return grid(s[k]["on"] == 1, s[k]["y"], s[k]["x"])
    bottom = CELLS[:, None] == 9
    return jnp.stack([
        cell(s["y"], s["x"]),
        cell(s["y"], s["x"] + 1 - 2 * s["right"]),
        objects("fb"),
        back(s["fish"]) | back(s["subs"]) | back(s["divers"]),
        objects("eb"),
        objects("fish"),
        objects("subs"),
        bottom & (CELLS < jnp.maximum(s["oxygen"], 0) * 10 // 200),
        bottom & (CELLS >= 9 - s["held"]) & (CELLS < 9),
        objects("divers"),
    ], -1).astype(jnp.float32)

# ---------------------------------------------------------------- SpaceInvaders
def invaders_reset(key):
    z = jnp.zeros((10, 10), bool)
    return dict(pos=i32(5), f=z, e=z, aliens=z.at[0:4, 2:8].set(True), dir=i32(-1), interval=i32(12),
                move_timer=i32(12), alien_shot=i32(10), ramp_index=i32(0), shot=i32(0))

def invaders_step(key, s, a):
    fire = (a == F) & (s["shot"] == 0)
    pos = lr(s["pos"], a)
    f = jnp.roll(jnp.where(fire, s["f"].at[9, s["pos"]].set(True), s["f"]), -1, 0).at[9].set(False)  # bullets up
    e = jnp.roll(s["e"], 1, 0).at[0].set(False)  # enemy bullets down
    al = s["aliens"]
    term = e[9, pos] | al[9, pos]
    move = s["move_timer"] == 0  # the aliens step sideways, or down and turn at an edge
    turn = move & ((al[:, 0].any() & (s["dir"] < 0)) | (al[:, 9].any() & (s["dir"] > 0)))
    term = term | (turn & al[9].any())
    side = jnp.where(s["dir"] < 0, jnp.roll(al, -1, 1), jnp.roll(al, 1, 1))
    al = jnp.where(turn, jnp.roll(al, 1, 0), jnp.where(move, side, al))
    term = term | al[9, pos]
    shoot = s["alien_shot"] == 0  # the lowest alien of the nearest column fires
    col = jnp.argmin(jnp.where(al.any(0), 10 * jnp.abs(jnp.arange(10) - pos) + jnp.arange(10), 1000))
    e = jnp.where(shoot, e.at[jnp.argmax(jnp.where(al[:, col], jnp.arange(10), -1)), col].set(True), e)
    kill = al & f
    al, f = al & ~kill, f & ~kill
    empty = ~al.any()  # a new wave, faster
    ramp = empty & (s["interval"] > 6)
    return dict(
        pos=pos,
        f=f,
        e=e,
        aliens=al | (empty & jnp.zeros((10, 10), bool).at[0:4, 2:8].set(True)),
        dir=jnp.where(turn, -s["dir"], s["dir"]),
        interval=s["interval"] - ramp,
        move_timer=jnp.where(move, jnp.minimum(s["aliens"].sum(), s["interval"]), s["move_timer"]) - 1,
        alien_shot=jnp.where(shoot, 10, s["alien_shot"]) - 1,
        ramp_index=s["ramp_index"] + ramp,
        shot=down(jnp.where(fire, 5, s["shot"])),
    ), kill.sum(), term

def invaders_obs(s):  # cannon, alien, alien moving left, alien moving right, friendly bullet, enemy bullet
    al, left = s["aliens"], s["dir"] < 0
    cannon = jnp.zeros((10, 10), bool).at[9, s["pos"]].set(True)
    return jnp.stack([cannon, al, al & left, al & ~left, s["f"], s["e"]], -1).astype(jnp.float32)

# ---------------------------------------------------------------- batched interface
RESETS = (asterix_reset, freeway_reset, seaquest_reset, invaders_reset)
STEPS = (asterix_step, freeway_step, seaquest_step, invaders_step)
OBSERVE = (asterix_obs, freeway_obs, seaquest_obs, invaders_obs)

def reset(key, game, n):  # n envs of one game (index or name)
    game = GAMES.index(game) if isinstance(game, str) else game
    keys = jax.random.split(key, n)
    return dict(game=jnp.zeros(n, jnp.int32) + game, time=jnp.zeros(n, jnp.int32),
                **{g: jax.vmap(f)(keys) for g, f in zip(GAMES, RESETS)})

def dispatch(branch, s, game):  # a static game runs only its branch; else lax.switch (a select under vmap)
    if game is not None:
        return branch(game)(s)
    return jax.lax.switch(s["game"][0], [branch(i) for i in range(len(GAMES))], s)

def step(key, s, action, game=None):  # -> (state, reward, done), done envs reset
    ks, kr = jax.random.split(key)
    n = action.shape[0]
    def branch(i):
        def f(s):
            new, r, term = jax.vmap(STEPS[i])(jax.random.split(ks, n), s[GAMES[i]], action)
            time = s["time"] + 1
            done = term | (time >= LIMIT[i])
            fresh = jax.vmap(RESETS[i])(jax.random.split(kr, n))
            new = jax.tree.map(lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), fresh, new)
            return s | {GAMES[i]: new, "time": jnp.where(done, 0, time)}, 1.0 * r, done
        return f
    return dispatch(branch, s, game)

def obs(s, game=None):  # float32 [n, OBS]
    def branch(i):
        def f(s):
            o = jax.vmap(OBSERVE[i])(s[GAMES[i]])
            return jnp.pad(o, ((0, 0), (0, 0), (0, 0), (0, C - o.shape[-1]))).reshape(len(o), OBS)
        return f
    return dispatch(branch, s, game)

def valid(s):  # bool [n, A]: the game's minimal action set
    return VALID[s["game"]]
