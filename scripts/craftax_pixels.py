"""Craftax-Classic with pixel observations (Matthews et al. 2024), an env module for mini.use_env and rule_craftax.py.

Runs the craftax package's game code (1.5.0, as our PPO in mbrl_flow): craftax_step, is_game_over, generate_world and
the 63x63 agent renderer, exactly as its CraftaxClassicPixelsEnvNoAutoReset does, without that class's gymnax import.
Default env params: an episode ends at death, in lava, or after 10000 steps; reward +1 per new achievement and
+-0.1 per health point. Observation: the frame as uint8 (rint of the renderer's 0..255, as the PPO's to_u8), flattened,
then one byte: the previous action (0 at an episode start), which mini's impala core feeds to its GRU, as the PPO does.
State: dict of the craftax EnvState, the frame, the previous action, and the achievements of the episodes that ended at
the last step (none elsewhere), for logs. Needs craftax and its imports (matplotlib, imageio) on the path.
"""

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.game_logic import craftax_step, is_game_over
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_classic.world_gen import generate_world

SIDE, A, ACHIEVEMENTS = 63, 17, 22
OBS = SIDE * SIDE * 3 + 1
GAMES = ("classic",)
PARAMS, STATIC = EnvParams(), StaticEnvParams()

def frame(env):  # uint8 [SIDE * SIDE * 3]
    pixels = render_craftax_pixels(env, BLOCK_PIXEL_SIZE_AGENT)
    return jnp.clip(jnp.rint(pixels), 0, 255).astype(jnp.uint8).reshape(-1)

def new_world(key):
    env = generate_world(key, PARAMS, STATIC)
    return env, frame(env)

def reset(key, game, n):  # n new worlds (one game)
    env, f = jax.vmap(new_world)(jax.random.split(key, n))
    return dict(env=env, frame=f, prev=jnp.zeros(n, jnp.int32), ended=jnp.zeros((n, ACHIEVEMENTS), bool))

def step(key, state, action):  # -> (state, reward, done); done envs start in a new world
    n, (key_step, key_reset) = len(action), jax.random.split(key)
    def one(k, env, a):
        env, reward = craftax_step(k, env, a, PARAMS, STATIC)
        return env, frame(env), reward, is_game_over(env, PARAMS)
    env, f, reward, done = jax.vmap(one)(jax.random.split(key_step, n), state["env"], action)
    ended = done[:, None] & env.achievements
    def fresh(old):  # worlds are made only when an episode ends (under vmap: always, then selected)
        def pick(new, x):
            return jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), new, x)
        return jax.tree.map(pick, jax.vmap(new_world)(jax.random.split(key_reset, n)), old)
    env, f = jax.lax.cond(done.any(), fresh, lambda old: old, (env, f))
    return dict(env=env, frame=f, prev=jnp.where(done, 0, action), ended=ended), reward, done

def obs(state):  # uint8 [n, OBS]
    return jnp.concatenate([state["frame"], state["prev"].astype(jnp.uint8)[:, None]], -1)

def valid(state):  # all 17 actions are legal
    return jnp.ones((len(state["prev"]), A), bool)
