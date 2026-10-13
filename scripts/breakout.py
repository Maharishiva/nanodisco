"""MinAtar Breakout in pure JAX, a dependency-free port of gymnax's Breakout-MinAtar.

Same dynamics, reward (+1 per brick), termination (ball missed or 1000 steps),
10x10x4 observation and minimal action set ['n', 'l', 'r'] as
gymnax/environments/minatar/breakout.py. State is a dict of arrays.
"""

import jax
import jax.numpy as jnp

OBS = 400
UP_DOWN, LR_FLIP, PADDLE = jnp.array([3, 2, 1, 0]), jnp.array([1, 0, 3, 2]), jnp.array([2, 3, 0, 1])

def reset(key):
    start = jax.random.choice(key, jnp.array([0, 1]), shape=())
    x = jnp.array([0, 9])[start]
    return dict(
        ball_y=jnp.array(3),
        ball_x=x,
        ball_dir=jnp.array([2, 3])[start],
        pos=jnp.array(4),
        brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
        strike=jnp.array(False),
        last_y=jnp.array(3),
        last_x=x,
        time=jnp.array(0),
        terminal=jnp.array(False),
    )

def obs(s):
    o = jnp.zeros((10, 10, 4))
    o = o.at[9, s["pos"], 0].set(1).at[s["ball_y"], s["ball_x"], 1].set(1).at[s["last_y"], s["last_x"], 2].set(1)
    return o.at[:, :, 3].set(s["brick_map"]).reshape(-1)

def step(s, action):  # no auto-reset; returns (state, reward, done)
    a = jnp.array([0, 1, 3])[action]
    pos = jnp.where(a == 1, jnp.maximum(0, s["pos"] - 1), jnp.where(a == 3, jnp.minimum(9, s["pos"] + 1), s["pos"]))
    last_x, last_y, d = s["ball_x"], s["ball_y"], s["ball_dir"]
    x, y = s["ball_x"] + jnp.array([-1, 1, 1, -1])[d], s["ball_y"] + jnp.array([-1, -1, 1, 1])[d]
    wall = (x < 0) | (x > 9)
    x, d = jnp.where(wall, 9 * (x > 9), x), jnp.where(wall, LR_FLIP[d], d)
    top = y < 0
    y, d = jnp.where(top, 0, y), jnp.where(top, UP_DOWN[d], d)
    toggle = ~top & (s["brick_map"][y, x] == 1)
    hit = ~s["strike"] & toggle
    bricks = jnp.where(hit, s["brick_map"].at[y, x].set(0), s["brick_map"])
    y, d = jnp.where(hit, last_y, y), jnp.where(hit, UP_DOWN[d], d)
    bottom = ~toggle & (y == 9)
    bricks = jnp.where(bottom & (jnp.count_nonzero(bricks) == 0), bricks.at[1:4, :].set(1), bricks)
    r1 = bottom & (s["ball_x"] == pos)  # gymnax compares the old ball column with the new paddle position
    y, d = jnp.where(r1, last_y, y), jnp.where(r1, UP_DOWN[d], d)
    r2 = bottom & ~r1 & (x == pos)
    y, d = jnp.where(r2, last_y, y), jnp.where(r2, PADDLE[d], d)
    time = s["time"] + 1
    done = (time >= 1000) | (bottom & ~r1 & ~r2)
    return (
        dict(
            ball_y=y,
            ball_x=x,
            ball_dir=d,
            pos=pos,
            brick_map=bricks,
            strike=toggle,
            last_y=last_y,
            last_x=last_x,
            time=time,
            terminal=done,
        ),
        1.0 * hit,
        done,
    )
