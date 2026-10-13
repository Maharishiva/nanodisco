"""miniDisco v3: a learned RL update rule on Catch, meta-trained the way it is evaluated.

Eight agents (2x256 MLP + 256 GRU, Adam) learn their policy and y/z predictions only by
distilling KL targets from a 256-unit reverse-GRU rule (eta). As in Disco103, the rule also
reads value, Q and advantage estimates from the agent's own Q head, trained by a fixed
Retrace loss. Every agent update is as in the evaluation (Disco103's paper setting): 8 envs x
29 new steps go into a replay buffer, and the agent learns from 96 trajectories, 90% of them
replayed. A meta-step backpropagates through 10 such updates (MixFlow-MG, Kemaev et al. 2025),
then scores the adapted agent on 48 new trajectories with a policy gradient whose advantages
come from a V-trace meta-critic. Each agent's meta-gradient goes through its own clipped Adam
and the steps are averaged. As in the paper, an agent restarts when its lifetime is over, and a
new lifetime is drawn with probability proportional to 1/lifetime, so every lifetime category gets
the same share of experience. Meta-training is one jit.
The metric is the training return of the agents. history/mini_v2.py is the on-policy version.
Faster variants, by switches (defaults are v3): AGENT_CORE (gru | mlp | gdn), RULE_CORE (gru | attn |
gdn), ASSOC (returns by associative scans) and UNROLL (True: no loops left in collect and the scans).
AGENT_CORE impala is for pixel envs (scripts/craftax_pixels.py): the IMPALA CNN + GRU of our Craftax PPO.
use_slots(n): every agent acts through n slots, a random number of them live and mapped to the game's actions in random
order (extra ones copy an action or are no-ops), so the rule meets many action counts (Craftax has 17).

Names: params = agent weights, eta = rule weights, critic = meta-critic weights,
rollout = dict of arrays from collect(): s (T+1 observations or env states),
first (T+1 episode-start flags), a, r, done, mu (behaviour logits) over T steps,
and h0 (agent memory at the start of the rollout). Rollouts are time-major; the
replay buffer stores them batch-major.
"""

import functools
import sys

import jax
import jax.numpy as jnp
import optax

A, D, W, T = 3, 16, 5, 4  # actions, y/z size, board width, Catch episode length
B, ROLL, INNER, AGENTS = 8, 29, 10, 8  # envs per agent, steps per rollout, agent updates per meta-step, agents
BATCH, REPLAY, BUFFER = 96, 0.9, 400_000  # trajectories per update, replayed share of them, buffer size (transitions)
UTD = 1  # agent updates per collected rollout (replay ratio), each on its own sampled batch
VALID = 6  # rollouts that score the adapted agent: 6 x 8 = 48 trajectories, as in the paper
HID, RULE_HID, EMB = 256, 256, 16  # agent width, rule GRU width, per-action embedding width
GAMMA, LAM = 0.99, 0.95
USE_Q = True  # False: no Q head, and the rule sees only rewards, done flags, policies and the learned y/z
RETRACE = True  # False: v2's uncorrected Q(lambda) targets, as the rules in results/catch_gpu_20261004 were trained
AGENT_LR, CRITIC_LR, META_LR = 5e-4, 1e-3, 1e-3
META_ENT = 1e-2  # entropy bonus on the adapted policy in the meta-objective
META_SHARED = False  # True: v2's single Adam on the agents' mean meta-gradient, no clip; "clip": each agent's
# meta-gradient clipped to norm 1 first (8 noisy agents: about sqrt(8) larger useful steps than 8 Adams)
UNROLL = 1  # time steps per loop iteration in collect and the agent and rule scans: >1 runs faster, compiles slower
AGENT_CORE = "gru"  # agent memory: gru | mlp (no memory, a third dense layer) | gdn (Gated DeltaNet-2, residual)
# | impala (CNN torso in place of l1 and l2, a GRU that also reads the previous action, a layer after it)
CNN, CNN_FC, PIXELS = (64, 128, 128), 512, (63, 63, 3)  # impala: CNN block channels, fc width, frame shape
CNN_DTYPE = jnp.bfloat16  # impala: compute type of the convolutions and fc (weights stay float32), as our PPO
CNN_REMAT = False  # impala: the backward pass recomputes each CNN block (stores 0.25 of 3.3 MB per frame, +1 forward)
CNN_STEPS = 0  # impala: time steps of a rollout per CNN pass (0: all; a few: less memory for large batches)
RULE_CORE = "gru"  # rule's reverse pass over time: gru | attn (reverse-causal attention) | gdn (reverse GDN-2)
ASSOC = False  # True: returns, Retrace and V-trace by jax.lax.associative_scan instead of a loop
RULE_DIM, RULE_LAYERS, HEADS = 128, 1, 4  # attn/gdn rule width and blocks, attention heads
RULE_LOGP = False  # True: the rule also reads log pi and log mu per action (log-softmax, floored at -10)
RULE_LEGAL = False  # True: the rule's reductions over actions (embedding mean, z max, Q norm) skip illegal actions
ANCHOR = 0.0  # a > 0: pi target = a * the agent's log-policy (stopped) + the rule's output; a = 1 never settles
LOGP_GRAD = False  # True: meta-gradients also flow through the rule's log pi input (agent updates do not)
REMAT = False  # True: the meta-gradient recomputes each agent update's forward pass instead of storing it (less memory)
GDN_K, GDN_HEADS, GDN_LAM, GDN_CHUNK = 32, 2, 3.0, 16  # gdn head size, agent heads, max -log decay per step, chunk
LIFETIMES = (50, 125, 250, 500)  # meta-steps (paper: 20M, 50M, 100M, 200M frames); () = agents never restart
ACT_MAX, GAME_A = 0, A  # use_slots(n): agents act through n slots, mapped per agent to the game's GAME_A actions
ACT_PAD = "mix"  # live slots past the game's actions: copy (a random game action), noop (NOOP), mix (either, per slot)
ACT_REDRAW = 0.0  # chance per agent update that the agent's slot map is drawn again: its action set changes in life
ACT_PLAIN = 0.0  # share of slot maps with no extra live slots: only the game's actions, in random order
OBS = 2 * W * (T + 1)
STEM, STEM_STEPS = 256, 1  # frame stem output size (agent input for packed 64x64 frames); time steps per stem pass
STEM_REMAT = True  # the backward pass recomputes the stem's one-hot activations instead of storing them (less memory)
ENV, ENVS = "catch", {}  # the meta-training env; env modules plugged in by use_env
sm, lsm, stop = jax.nn.softmax, jax.nn.log_softmax, jax.lax.stop_gradient

def take(x, a):  # x[..., a, :]
    return jnp.take_along_axis(x, a[..., None, None], -2)[..., 0, :]

def pick(x, a):  # x[..., a], keeps the axis
    return jnp.take_along_axis(x, a[..., None], -1)

def kl(p, q):  # KL(teacher || student) from logits
    return (sm(p) * (lsm(p) - lsm(q))).sum(-1)

def entropy(logits):
    return -(sm(logits) * lsm(logits)).sum(-1).mean()

def signlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def norm(x, legal=None):  # legal: normalize with the statistics of the legal entries only
    if legal is None:
        return (x - x.mean()) / (x.std() + 1e-6)
    n = legal.sum()
    mean = (x * legal).sum() / n
    return (x - mean) / (jnp.sqrt(((x - mean) ** 2 * legal).sum() / n) + 1e-6)

def drop_last(tree):  # outputs on T+1 observations -> the T steps that have actions
    return jax.tree.map(lambda x: x[:-1], tree)

# ---------------------------------------------------------------- environments
def env_step(state, action, env):  # state = (t, goal column, player column)
    t, goal = state[..., 0], state[..., 1]
    player = jnp.clip(state[..., 2] + action - 1, 0, W - 1)
    hit, timeout = player == goal, t == T - 1
    if env == "catch":  # +1 or -1 when the ball lands
        reward, done = jnp.where(timeout, 2.0 * hit - 1, 0.0), timeout
    else:  # reach: +1 on reaching the goal
        reward, done = 1.0 * hit, hit | timeout
    return jnp.stack([t + 1, goal, player], -1), reward, done

def reset(key, env, n=None):  # catch: ball in any column; reach: goal at either edge; n=None: B envs
    n = B if n is None else n  # read B at call time, so a B set after import (--set B=32) is used
    goal = jax.random.randint(key, (n,), 0, W) if env == "catch" else (W - 1) * jax.random.randint(key, (n,), 0, 2)
    return jnp.stack([jnp.zeros_like(goal), goal, jnp.full_like(goal, W // 2)], -1)

def obs(state, env):  # 5x5 board, two channels: ball/goal row and paddle/player row
    if env in ENVS:  # rollouts of env modules hold observations
        return state.astype(jnp.float32)
    top, bottom = (state[..., 0], T) if env == "catch" else (1, 1)
    def cell(row, col):
        return jax.nn.one_hot(row * W + col, (T + 1) * W)
    return jnp.concatenate([cell(top, state[..., 1]), cell(bottom, state[..., 2])], -1)

def use_env(name, module):  # plug in an env module and make it the meta-training env
    # module: OBS, A, GAMES, reset(key, game, n) -> state, step(key, state, action) -> (state, reward, done) with
    # auto-reset, obs(state) -> [n, OBS], valid(state) -> bool [n, A]; FRAME = True: obs are packed 64x64 frames
    global ENV, OBS, A
    ENVS[name], ENV, OBS, A = module, name, STEM if getattr(module, "FRAME", False) else module.OBS, module.A

def observe(state, env, slots=None):  # what the agent reads, and its legal actions (None: all)
    s, valid = (ENVS[env].obs(state), ENVS[env].valid(state)) if env in ENVS else (state, None)
    if slots is None:
        return s, valid
    live = jnp.broadcast_to(slots >= 0, (len(s), ACT_MAX))  # a slot is legal if it is live and its game action is
    if valid is None:
        return s, live
    return s, live & jnp.take_along_axis(valid, jnp.broadcast_to(jnp.maximum(slots, 0), live.shape), -1)

def use_slots(n):  # agents get n action slots; call after use_env
    global ACT_MAX, GAME_A, A
    ACT_MAX, GAME_A, A = n, A, n

def draw_slots(key, env):  # slot -> game action, -1: dead slot. A random number (GAME_A to ACT_MAX) of live slots in
    # random order: GAME_A of them hold each game action once, the others copy a random game action or the no-op
    k = jax.random.split(key, 4)
    live = jax.random.randint(k[0], (), GAME_A, ACT_MAX + 1)
    live = jnp.where(jax.random.bernoulli(jax.random.fold_in(k[0], 1), ACT_PLAIN), GAME_A, live)
    pad = jax.random.randint(k[1], (ACT_MAX,), 0, GAME_A)
    noop = 1 if env not in ENVS else getattr(ENVS[env], "NOOP", None)  # catch, reach: 1 = stay
    if ACT_PAD != "copy" and noop is not None:
        pad = jnp.where(ACT_PAD == "noop" or jax.random.bernoulli(k[2], 0.5, (ACT_MAX,)), noop, pad)
    i = jnp.arange(ACT_MAX)
    return jax.random.permutation(k[3], jnp.where(i < GAME_A, i, jnp.where(i < live, pad, -1)))

def env_reset(key, env, game=0):  # B envs of one game
    return ENVS[env].reset(key, game, B) if env in ENVS else reset(key, env)

def agent_games(n, env, key=None):  # agent i plays game i mod the number of games; with a key, random games
    count = len(ENVS[env].GAMES) if env in ENVS else 1
    return jnp.arange(n) % count if key is None else jax.random.randint(key, (n,), 0, count)

def masked(logits, valid):  # illegal actions get no probability
    return logits if valid is None else jnp.where(valid, logits, -1e9)

# ---------------------------------------------------------------- layers
def dense(key, n_in, n_out, scale=None):  # LeCun-normal weights unless a scale is given
    return dict(w=(scale or n_in**-0.5) * jax.random.normal(key, (n_in, n_out)), b=jnp.zeros(n_out))

def apply(layer, x):
    return x @ layer["w"] + layer["b"]

def init_gru(key, n_in, n):
    key_x, key_h = jax.random.split(key)
    return dict(x=dense(key_x, n_in, 3 * n), h=dense(key_h, n, 3 * n))

def gru(layer, h, x):
    return gru_cell(layer["h"], h, apply(layer["x"], x))

def gru_cell(layer_h, h, gx):  # gx = apply(layer["x"], x): loops compute it for all steps at once, outside the loop
    (xr, xz, xn), (hr, hz, hn) = jnp.split(gx, 3, -1), jnp.split(apply(layer_h, h), 3, -1)
    r, z = jax.nn.sigmoid(xr + hr), jax.nn.sigmoid(xz + hz)
    return (1 - z) * jnp.tanh(xn + r * hn) + z * h

def rms(x):  # RMS norm over the last axis, no gain
    return x * jax.lax.rsqrt((x**2).mean(-1, keepdims=True) + 1e-6)

def init_gdn(key, n, heads):  # one Gated DeltaNet-2 layer (arXiv:2605.22791), without the short convolutions
    key_in, key_out, key_decay = jax.random.split(key, 3)
    proj, hk = dense(key_in, n, 7 * heads * GDN_K), heads * GDN_K  # q, k, v, output gate, decay, erase, write
    decay = jnp.log(jnp.expm1(jax.random.uniform(key_decay, (hk,), minval=0.01, maxval=0.5)))  # -log decay at init
    proj["b"] = proj["b"].at[4 * hk : 5 * hk].set(decay)
    return dict(proj=proj, out=dense(key_out, hk, n))

def gdn_gates(layer, x):  # per head [..., heads, GDN_K]: unit q and k, v, output gate, log decay, erase and write gates
    q, k, v, gate, a, b, w = (p.reshape(p.shape[:-1] + (-1, GDN_K)) for p in jnp.split(apply(layer["proj"], x), 7, -1))
    def unit(x):
        return x * jax.lax.rsqrt((x**2).sum(-1, keepdims=True) + 1e-6)
    log_a = -GDN_LAM * jnp.tanh(jax.nn.softplus(a) / GDN_LAM)  # in (-GDN_LAM, 0]: keeps the chunk exponents finite
    return unit(jax.nn.silu(q)), unit(jax.nn.silu(k)), jax.nn.silu(v), gate, log_a, jax.nn.sigmoid(b), jax.nn.sigmoid(w)

def gdn_out(layer, o, gate):  # per-head norm, output gate, projection
    o = rms(o) * jax.nn.silu(gate)
    return apply(layer["out"], o.reshape(o.shape[:-2] + (-1,)))

def gdn_step(layer, S, x):  # one step: S <- (I - k (b k)^T) diag(a) S + k (w v)^T, o = S^T q; S [..., heads, K, K]
    q, k, v, gate, log_a, b, w = gdn_gates(layer, x)
    a = jnp.exp(log_a)
    uo = jnp.einsum("...ik,...kv->...iv", jnp.stack([b * k * a, q * a], -2), S)  # (b k a)^T S and (q a)^T S at once
    u = w * v - uo[..., 0, :]
    S = a[..., None] * S + k[..., None] * u[..., None, :]
    o = uo[..., 1, :] + (q * k).sum(-1, keepdims=True) * u  # = S^T q for the new S
    # the barrier stops XLA GPU fusion from merging the steps of an unrolled loop (frozen update compile 1000 s -> 60 s)
    return jax.lax.optimization_barrier(S), gdn_out(layer, o, gate)

def solve(A, rhs):  # (I + A)^-1 rhs for strictly lower-triangular A [C, C]: (I - A)(I + A^2)(I + A^4)...
    x, p = rhs - A @ rhs, A
    for _ in range(max(0, (A.shape[-1] - 1).bit_length() - 1)):
        p = p @ p
        x = x + p @ x
    return x

def gdn_seq(layer, S, x, first):  # gdn_step over a time-major x [L, ...] from state S, in chunks (chunkwise WY form)
    q, k, v, gate, log_a, b, w = gdn_gates(layer, x)  # first: the state resets before this step
    f = jnp.broadcast_to(first[..., None, None], log_a.shape[:-1] + (1,))
    n, pad = len(x), -len(x) % GDN_CHUNK
    def chunks(t):  # [L, ..., heads, d] -> [chunks, ..., heads, GDN_CHUNK, d]
        t = jnp.pad(t, [(0, pad)] + [(0, 0)] * (t.ndim - 1))
        return jnp.moveaxis(t.reshape((-1, GDN_CHUNK) + t.shape[1:]), 1, -2)
    ti = jnp.arange(GDN_CHUNK)
    def chunk(S, inputs):  # unrolled (few chunks, 1.3x faster); backward recomputes it (meta-step memory -20-30%)
        q, k, v, log_a, b, w, f = inputs
        seg = jnp.cumsum(f, -2)  # episode index in the chunk, [..., C, 1]; 0 continues from S
        same, live, last = seg == jnp.swapaxes(seg, -1, -2), seg == 0, seg == seg[..., -1:, :]
        g = jnp.cumsum(log_a, -2)
        decay, keys = jnp.exp(g), (k * jnp.exp(-g)).swapaxes(-1, -2)  # e^{g_t - g_s} = e^{g_t} e^{-g_s}
        A = jnp.where(same & (ti[:, None] > ti), (b * k * decay) @ keys, 0.0)
        P = jnp.where(same & (ti[:, None] >= ti), (q * decay) @ keys, 0.0)
        R, U = jnp.split(solve(A, jnp.concatenate([live * decay * b * k, w * v], -1)), 2, -1)
        U = U - R @ S
        o = (live * q * decay) @ S + P @ U
        end = g[..., -1:, :]
        S = live[..., -1:, :] * jnp.exp(end).swapaxes(-1, -2) * S + (last * jnp.exp(end - g) * k).swapaxes(-1, -2) @ U
        return S, o
    o = jax.lax.scan(jax.checkpoint(chunk), S, tuple(chunks(t) for t in (q, k, v, log_a, b, w, f)), unroll=True)[1]
    o = jnp.moveaxis(o, -2, 1).reshape((-1,) + o.shape[1:-2] + o.shape[-1:])[:n]
    return gdn_out(layer, o, gate)

# ---------------------------------------------------------------- frame stem (packed 64x64 frames, 16 colours)
def init_stem(key):  # 4x4 patches of one-hot colours -> 32 channels; 2x2 of those -> 64; 8x8x64 -> STEM
    keys = jax.random.split(key, 3)
    patch, mix = dense(keys[0], 16 * 16, 32, 0.25), dense(keys[1], 4 * 32, 64)
    return dict(patch=patch, mix=mix, out=dense(keys[2], 64 * 64, STEM))

def stem(p, s):  # s: uint8 [..., 2048], pixel 2i in the low nibble of byte i, rows of 64 pixels
    lead = s.shape[:-1]
    pixels = jnp.stack([s & 15, s >> 4], -1).reshape(lead + (16, 4, 16, 4))
    patches = jnp.swapaxes(pixels, -3, -2).reshape(lead + (256, 16))  # 16x16 patches of 4x4 pixels
    one_hot = jax.nn.one_hot(patches, 16, dtype=jnp.bfloat16).reshape(lead + (256, 256))  # exact in bf16
    w = p["patch"]["w"].astype(jnp.bfloat16)
    x = jax.nn.relu(jnp.dot(one_hot, w, preferred_element_type=jnp.float32) + p["patch"]["b"])
    x = jnp.swapaxes(x.reshape(lead + (8, 2, 8, 2, 32)), -4, -3).reshape(lead + (64, 128))  # 8x8 groups of 2x2 patches
    return jax.nn.relu(apply(p["out"], jax.nn.relu(apply(p["mix"], x)).reshape(lead + (64 * 64,))))

def with_stem(params, key, env):  # agents and critics of frame envs read frames through their own stem
    return params | dict(stem=init_stem(key)) if getattr(ENVS.get(env), "FRAME", False) else params

def embed(params, state, env):  # agent and critic input; the stem reads a rollout's frames STEM_STEPS steps at a time
    if "stem" not in params:
        return obs(state, env)
    f = jax.checkpoint(stem) if STEM_REMAT else stem
    if state.ndim > 2:  # [T+1, batch, 2048]: a loop keeps the one-hot patches of a few steps in memory, not of all
        return jax.lax.map(lambda s: f(params["stem"], s), state, batch_size=STEM_STEPS)
    return f(params["stem"], state)

# ---------------------------------------------------------------- IMPALA CNN (AGENT_CORE impala, pixel frames)
def init_impala(key):  # per block: a conv, then 2 residual blocks of 2 convs; fc; GRU on [fc, previous action]; l3
    keys, blocks, c, side = jax.random.split(key, 4), [], PIXELS[-1], PIXELS[0]
    for i, ch in enumerate(CNN):  # a conv is a dense layer on 3x3 patches: LeCun-normal on its fan-in
        k = jax.random.split(jax.random.fold_in(keys[0], i), 5)
        blocks.append([dense(k[0], 9 * c, ch)] + [dense(x, 9 * ch, ch) for x in k[1:]])
        c, side = ch, -(-side // 2)
    return dict(cnn=blocks, fc=dense(keys[1], side * side * c, CNN_FC), gru=init_gru(keys[2], CNN_FC + A, HID),
                l3=dense(keys[3], CNN_FC + HID, CNN_FC))

def conv(layer, x):  # 3x3, stride 1, SAME, NHWC, in CNN_DTYPE
    w = layer["w"].reshape(3, 3, -1, layer["w"].shape[-1]).astype(CNN_DTYPE)
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + layer["b"].astype(CNN_DTYPE)

def max_pool(x):  # 3x3, stride 2, SAME, NHWC: reduce_window max, whose gradient goes to the first maximum of each
    # window in row-major order (select_and_scatter); here as 9 strided slices, so the gradient is a select (2x faster)
    (lo_h, hi_h), (lo_w, hi_w) = jax.lax.padtype_to_pads(x.shape[1:3], (3, 3), (2, 2), "SAME")
    x = jnp.pad(x, ((0, 0), (lo_h, hi_h), (lo_w, hi_w), (0, 0)), constant_values=-jnp.inf)
    h, w = (x.shape[1] - 1) // 2, (x.shape[2] - 1) // 2
    taps = [x[:, i : i + 2 * h - 1 : 2, j : j + 2 * w - 1 : 2] for i in range(3) for j in range(3)]
    top = stop(functools.reduce(jnp.maximum, taps))
    out, taken = jnp.zeros_like(top), jnp.zeros(top.shape, bool)
    for t in taps:
        first = (t == top) & ~taken
        out, taken = out + jnp.where(first, t, 0.0), taken | first
    return out

def cnn_block(block, x):  # conv, 3x3 max pool with stride 2, two residual blocks
    x = max_pool(conv(block[0], x))
    for i in (1, 3):
        x = x + conv(block[i + 1], jax.nn.relu(conv(block[i], jax.nn.relu(x))))
    return x

def cnn(p, s):  # s [..., OBS]: a uint8 frame, then the previous action -> [..., CNN_FC + A]: features, action one-hot
    lead = s.shape[:-1]
    x = (s[..., :-1].reshape((-1,) + PIXELS) / 255.0).astype(CNN_DTYPE)
    for block in p["cnn"]:
        x = (jax.checkpoint(cnn_block) if CNN_REMAT else cnn_block)(block, x)
    x = jax.nn.relu(x).reshape(lead + (-1,))
    fc = jnp.dot(x, p["fc"]["w"].astype(CNN_DTYPE), preferred_element_type=jnp.float32) + p["fc"]["b"]
    return jnp.concatenate([jax.nn.relu(fc), jax.nn.one_hot(s[..., -1].astype(jnp.int32), A)], -1)

def after_gru(p, x, g):  # the layer on [CNN features, GRU state] that the heads read
    return jax.nn.relu(apply(p["l3"], jnp.concatenate([x[..., :CNN_FC], g], -1)))

# ---------------------------------------------------------------- agent
def init_agent(key):  # the core's weights sit under the name of AGENT_CORE (impala: all but the heads)
    keys = jax.random.split(key, 7)
    if AGENT_CORE == "impala":
        body, width = {"impala": init_impala(keys[2])}, CNN_FC
    else:
        cores = dict(gru=init_gru, mlp=dense, gdn=lambda k, n, _: init_gdn(k, n, GDN_HEADS))
        core = cores[AGENT_CORE](keys[2], HID, HID)
        body, width = {"l1": dense(keys[0], OBS, HID), "l2": dense(keys[1], HID, HID), AGENT_CORE: core}, HID
    return body | {
        "pi": dense(keys[3], width, A, 0.01),
        "y": dense(keys[4], width, D, 0.01),
        "z": dense(keys[5], width, A * D, 0.01),
        "q": dense(keys[6], width, A, 0.01),
    }

def memory_size():  # agent memory per env: the GRU state, a dummy (mlp), or the flat GDN-2 state matrices
    return dict(gru=HID, mlp=1, gdn=GDN_HEADS * GDN_K**2, impala=HID)[AGENT_CORE]

def torso(params, state, env):  # input -> features, on any leading axes
    if AGENT_CORE == "impala":
        s = embed(params, state, env)
        if CNN_STEPS and s.ndim > 2:  # [T+1, batch, OBS]
            return jax.lax.map(lambda x: cnn(params["impala"], x), s, batch_size=CNN_STEPS)
        return cnn(params["impala"], s)
    return jax.nn.relu(apply(params["l2"], jax.nn.relu(apply(params["l1"], embed(params, state, env)))))

def heads(params, h):
    z = apply(params["z"], h).reshape(h.shape[:-1] + (A, D))
    return dict(pi=apply(params["pi"], h), y=apply(params["y"], h), z=z, q=apply(params["q"], h))

def agent_step(params, memory, state, first, env):  # one step of the agent; memory resets at episode starts
    x, memory = torso(params, state, env), memory * (1.0 - first)[..., None]
    if AGENT_CORE == "gru":
        memory = h = gru(params["gru"], memory, x)
    elif AGENT_CORE == "impala":
        memory = gru(params["impala"]["gru"], memory, x)
        h = after_gru(params["impala"], x, memory)
    elif AGENT_CORE == "mlp":
        h = jax.nn.relu(apply(params["mlp"], x))
    else:
        S, o = gdn_step(params["gdn"], memory.reshape(memory.shape[:-1] + (GDN_HEADS, GDN_K, GDN_K)), x)
        memory, h = S.reshape(memory.shape), x + o
    return memory, heads(params, h)

def unroll(params, rollout, env):  # agent outputs on all T+1 observations of a rollout, from its stored memory
    x, first, h0 = torso(params, rollout["s"], env), rollout["first"], rollout["h0"]
    if AGENT_CORE in ("gru", "impala"):  # only the recurrence is a loop
        layer = params["gru"] if AGENT_CORE == "gru" else params["impala"]["gru"]
        def step(h, inputs):
            gx, first = inputs
            h = gru_cell(layer["h"], h * (1.0 - first)[..., None], gx)
            return h, h
        h = jax.lax.scan(step, h0, (apply(layer["x"], x), first), unroll=UNROLL)[1]
        h = after_gru(params["impala"], x, h) if AGENT_CORE == "impala" else h
    elif AGENT_CORE == "mlp":
        h = jax.nn.relu(apply(params["mlp"], x))
    else:
        h = x + gdn_seq(params["gdn"], h0.reshape(h0.shape[:-1] + (GDN_HEADS, GDN_K, GDN_K)), x, first)
    outputs = heads(params, h)
    return outputs | dict(pi=masked(outputs["pi"], rollout.get("valid")))

def collect(params, carry, key, env, n=None):  # n steps (None: ROLL) in B envs; carry = (env state, memory, start flag)
    # and with use_slots the agent's slot map; rollouts hold slots (actions, legal actions, logits), not game actions
    n, slots = ROLL if n is None else n, carry[3] if len(carry) > 3 else None
    def step(carry, key):  # rollouts record env states (catch, reach) or observations and legal actions (env modules)
        state, memory, first = carry
        key_action, key_reset = jax.random.split(key)
        s, valid = observe(state, env, slots)
        memory, outputs = agent_step(params, memory, s, first, env)
        pi = masked(outputs["pi"], valid)
        action = jax.random.categorical(key_action, pi)
        game_action = action if slots is None else slots[action]
        if env in ENVS:  # env modules reset by themselves
            next_state, reward, done = ENVS[env].step(key_reset, state, game_action)
        else:
            next_state, reward, done = env_step(state, game_action, env)
            next_state = jnp.where(done[:, None], reset(key_reset, env, len(state)), next_state)
        record = dict(s=s, first=first, a=action, r=reward, done=1.0 * done, mu=pi)
        return (next_state, memory, 1.0 * done), record | ({} if valid is None else dict(valid=valid))
    (state, memory, first), rollout = jax.lax.scan(step, carry[:3], jax.random.split(key, n), unroll=UNROLL)
    s, valid = observe(state, env, slots)
    rollout["s"] = jnp.concatenate([rollout["s"], s[None]])  # the last observation, for bootstrapping
    if valid is not None:
        rollout["valid"] = jnp.concatenate([rollout["valid"], valid[None]])
    rollout["first"] = jnp.concatenate([rollout["first"], first[None]])
    rollout["h0"] = carry[1]
    return (state, memory, first) + carry[3:], jax.tree.map(stop, rollout)

def adam(params, opt_state, grads):  # Adam, clipped to 1 per element; no meta-gradient through the denominator
    m, v, n = opt_state
    m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, m, grads)
    v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * stop(g) ** 2, v, grads)
    n = n + 1
    def update(p, m, v):
        step = m / (1 - 0.9**n) / (jnp.sqrt(v / (1 - 0.999**n)) + 1e-8)
        return p - AGENT_LR * jnp.clip(step, -1.0, 1.0)
    return jax.tree.map(update, params, m, v), (m, v, n)

# ---------------------------------------------------------------- returns, and the agent's own Q head
def backward(a, b, end):  # G_t = a_t + b_t G_{t+1}, G_T = end, as an associative scan of the maps x -> a + b x
    a = a.at[-1].add(b[-1] * end)
    def compose(later, earlier):
        return earlier[0] + earlier[1] * later[0], earlier[1] * later[1]
    return jax.lax.associative_scan(compose, (a, b), reverse=True)[0]

def lambda_returns(r, discount, v_next, lam):  # G_t = r_t + discount_t ((1 - lam) v_{t+1} + lam G_{t+1})
    def back(g, inputs):
        r, discount, v = inputs
        g = r + discount * ((1 - lam) * v + lam * g)
        return g, g
    if ASSOC:
        return backward(r + discount * (1 - lam) * v_next, discount * lam, v_next[-1])
    return jax.lax.scan(back, v_next[-1], (r, discount, v_next), reverse=True, unroll=UNROLL)[1]

def importance(logits, rollout):  # pi(a) / mu(a) on the taken actions, stopped; logits for the T steps
    a = rollout["a"]
    return stop(jnp.exp(pick(lsm(logits), a) - pick(lsm(rollout["mu"]), a))[..., 0])

def retrace(r, discount, v_next, q_next, c_next):  # G_t = r_t + d_t (v_{t+1} + c_{t+1} (G_{t+1} - Q(s_{t+1}, a_{t+1})))
    def back(g, inputs):
        r, discount, v, q, c = inputs
        g = r + discount * (v + c * (g - q))
        return g, g
    if ASSOC:
        return backward(r + discount * (v_next - c_next * q_next), discount * c_next, 0.0)
    inputs = (r, discount, v_next, q_next, c_next)
    return jax.lax.scan(back, jnp.zeros_like(v_next[-1]), inputs, reverse=True, unroll=UNROLL)[1]

def vtrace(v, r, discount, rho, lam):  # V-trace targets and policy-gradient advantages (IMPALA), rho clipped at 1
    rho = jnp.minimum(rho, 1.0)
    delta = rho * (r + discount * v[1:] - v[:-1])
    def back(acc, inputs):
        delta, discount, c = inputs
        acc = delta + discount * c * acc
        return acc, acc
    if ASSOC:
        targets = v[:-1] + backward(delta, discount * lam * rho, 0.0)
    else:
        targets = v[:-1] + jax.lax.scan(back, jnp.zeros_like(v[-1]), (delta, discount, lam * rho), reverse=True)[1]
    targets_next = jnp.concatenate([targets[1:], v[-1:]])
    return targets, rho * (r + discount * targets_next - v[:-1])

def q_stats(outputs, rollout):  # Disco103-style value inputs, all stopped: v = pi . q, Q targets, advantages
    v = (sm(outputs["pi"]) * outputs["q"]).sum(-1)
    discount = GAMMA * (1 - rollout["done"])
    if RETRACE:  # traces cut by c = lambda min(1, pi/mu), and none past the last step
        q_taken = pick(outputs["q"][:-1], rollout["a"])[..., 0]
        c = LAM * jnp.minimum(1.0, importance(outputs["pi"][:-1], rollout))
        def next_step(x):
            return jnp.concatenate([x[1:], jnp.zeros_like(x[:1])])
        g = retrace(rollout["r"], discount, v[1:], next_step(q_taken), next_step(c))
    else:
        g = lambda_returns(rollout["r"], discount, v[1:], LAM)
    return stop(dict(v=v, g=g, adv=g - v[:-1], qv=outputs["q"] - v[..., None]))

# ---------------------------------------------------------------- the learned rule (eta)
def features(outputs, rollout, values):  # per-step and per-action rule inputs, following Disco103's input list
    pi, mu, a = stop(sm(outputs["pi"])), stop(sm(rollout["mu"])), rollout["a"]
    y, z = sm(outputs["y"]), sm(outputs["z"])
    legal = rollout.get("valid") if RULE_LEGAL else None  # T+1 steps
    pz = (z * pi[..., None]).sum(-2)
    zmax = (z if legal is None else jnp.where(legal[..., None], z, -jnp.inf)).max(-2)
    def col(x):
        return x[..., None]
    step = [pick(pi[:-1], a), pick(mu, a), col(signlog(rollout["r"])), col(1 - rollout["done"])]
    if USE_Q:
        v, adv = values["v"], values["adv"]
        step += [col(signlog(v[:-1])), col(signlog(v[1:])), col(signlog(adv)), col(norm(adv))]
    step += [y[:-1], y[1:], take(z[:-1], a), pz[:-1], pz[1:], zmax[:-1], zmax[1:]]
    per_action = [col(pi[:-1]), col(mu), z[:-1]]
    if USE_Q:
        q, qv = stop(outputs["q"][:-1]), values["qv"][:-1]
        per_action += [col(signlog(q)), col(signlog(qv)), col(norm(qv, None if legal is None else legal[:-1]))]
    per_action += [col(jax.nn.one_hot(a, A))]
    if RULE_LOGP:  # illegal actions read -10
        logp = jnp.maximum(lsm(outputs["pi"][:-1]), -10.0)
        per_action += [col(logp if LOGP_GRAD else stop(logp)), col(stop(jnp.maximum(lsm(rollout["mu"]), -10.0)))]
    return jnp.concatenate(step, -1), jnp.concatenate(per_action, -1)

def init_rule(key):  # the core's weights sit under the name of RULE_CORE
    keys, n_step, n_action = jax.random.split(key, 5), 4 * USE_Q + 4 + 7 * D, 3 * USE_Q + 3 + D + 2 * RULE_LOGP
    n_in, hid = n_step + 2 * EMB, RULE_HID if RULE_CORE == "gru" else RULE_DIM
    return {
        "emb": dense(keys[0], n_action, EMB),
        RULE_CORE: init_gru(keys[1], n_in, RULE_HID) if RULE_CORE == "gru" else init_blocks(keys[1], n_in),
        "pol": dense(keys[2], hid + EMB, EMB),
        "pi": dense(keys[3], EMB, 1, 0.01),
        "yz": dense(keys[4], hid, 2 * D, 0.01),
    }

def init_blocks(key, n_in):  # input layer, then RULE_LAYERS residual blocks: time mixing (attn or gdn), then an MLP
    keys = jax.random.split(key, 1 + 4 * RULE_LAYERS)
    def mixer(k1, k2):
        if RULE_CORE == "gdn":
            return init_gdn(k1, RULE_DIM, RULE_DIM // GDN_K)
        return dict(qkv=dense(k1, RULE_DIM, 3 * RULE_DIM), out=dense(k2, RULE_DIM, RULE_DIM))
    blocks = [dict(mix=mixer(*keys[4 * i + 1 : 4 * i + 3]), up=dense(keys[4 * i + 3], RULE_DIM, 2 * RULE_DIM),
                   down=dense(keys[4 * i + 4], 2 * RULE_DIM, RULE_DIM)) for i in range(RULE_LAYERS)]
    return dict(inp=dense(keys[0], n_in, RULE_DIM), blocks=blocks)

def attend(layer, x, done):  # every step attends to itself and the later steps of its episode; ALiBi distance bias
    q, k, v = (t.reshape(t.shape[:-1] + (HEADS, -1)) for t in jnp.split(apply(layer["qkv"], x), 3, -1))
    t, episode = jnp.arange(len(x)), jnp.cumsum(done, 0) - done  # episode index of every step
    ahead = t - t[:, None]  # [t, s]: s - t
    mask = (ahead >= 0)[..., None] & (episode[:, None] == episode[None])  # [t, s, ...]
    bias = -ahead[..., None] / 4.0 ** jnp.arange(HEADS)  # [t, s, heads]: slopes 1, 1/4, 1/16, ...
    logits = jnp.einsum("t...hd,s...hd->t...hs", q, k) / q.shape[-1] ** 0.5
    logits = logits + jnp.moveaxis(bias, 1, -1).reshape((len(x),) + (1,) * (x.ndim - 2) + bias.shape[-1:] + (len(x),))
    logits = jnp.where(jnp.moveaxis(mask, 1, -1)[..., None, :], logits, -1e30)
    o = jnp.einsum("t...hs,s...hd->t...hd", sm(logits), v)
    return apply(layer["out"], o.reshape(x.shape))

def blocks(core, x, done):  # RULE_LAYERS pre-norm residual blocks, each step sees only later steps of its episode
    h = apply(core["inp"], x)
    for block in core["blocks"]:
        if RULE_CORE == "gdn":  # reverse time, reset after every episode end
            S = jnp.zeros(h.shape[1:-1] + (RULE_DIM // GDN_K, GDN_K, GDN_K))
            h = h + gdn_seq(block["mix"], S, rms(h)[::-1], done[::-1])[::-1]
        else:
            h = h + attend(block["mix"], rms(h), done)
        h = h + apply(block["down"], jax.nn.relu(apply(block["up"], rms(h))))
    return rms(h)

def rule(eta, step, per_action, rollout):  # reverse pass over time (cut at episode ends); per-action policy head
    e = jax.nn.relu(apply(eta["emb"], per_action))
    if RULE_LEGAL and "valid" in rollout:  # mean over the legal actions only
        legal = rollout["valid"][:-1, ..., None]
        mean = (e * legal).sum(-2) / legal.sum(-2)
    else:
        mean = e.mean(-2)
    x = jnp.concatenate([step, mean, take(e, rollout["a"])], -1)
    if RULE_CORE == "gru":  # only the recurrence is a loop
        def cell(h, inputs):
            gx, done = inputs
            h = gru_cell(eta["gru"]["h"], h * (1 - done)[..., None], gx)
            return h, h
        h0, gx = jnp.zeros(x.shape[1:-1] + (RULE_HID,)), apply(eta["gru"]["x"], x)
        _, hs = jax.lax.scan(cell, h0, (gx, rollout["done"]), reverse=True, unroll=UNROLL)
    else:
        hs = blocks(eta[RULE_CORE], x, rollout["done"])
    w, n = eta["pol"]["w"], hs.shape[-1]  # pol on [hs, e] per action, without a copy of hs per action
    pol = (hs @ w[:n])[..., None, :] + e @ w[n:] + eta["pol"]["b"]
    y_target, z_target = jnp.split(apply(eta["yz"], hs), 2, -1)
    return apply(eta["pi"], jax.nn.relu(pol))[..., 0], y_target, z_target

def anchor(targets, outputs):  # ANCHOR a: pi target = a * the agent's log-policy (stopped) + the rule's output g, so
    # the policy keeps a memory and settles at log-probabilities g / (1 - a): a fixed point whose sharpness follows g
    return ((targets[0] + ANCHOR * stop(lsm(outputs["pi"][:-1])),) + targets[1:]) if ANCHOR else targets

# ---------------------------------------------------------------- inner update with a MixFlow-MG backward pass
def inner_loss(params, targets, rollout, env):  # KL to the rule's pi/y/z targets + TD loss for the Q head
    return outputs_loss(drop_last(unroll(params, rollout, env)), targets, rollout)

def outputs_loss(outputs, targets, rollout):  # inner_loss on the agent's outputs for the T steps
    a = rollout["a"]
    pi_target = masked(targets[0], drop_last(rollout.get("valid")))  # the rule's policy, on legal actions only
    loss = kl(pi_target, outputs["pi"]) + kl(targets[1], outputs["y"]) + kl(targets[2], take(outputs["z"], a))
    if USE_Q:  # fixed TD loss for the Q head; targets[3] holds its Q(lambda) targets
        loss += 0.5 * (pick(outputs["q"], a)[..., 0] - targets[3]) ** 2
    return loss.mean()

@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def inner_grad(params, targets, rollout, env):
    return jax.grad(inner_loss)(params, targets, rollout, env)

def inner_grad_fwd(params, targets, rollout, env):
    return inner_grad(params, targets, rollout, env), (params, targets, rollout)

def inner_grad_bwd(env, saved, v):  # v^T Hessian = Hessian v (symmetric): one forward-over-reverse jvp
    params, targets, rollout = saved
    def grad_both(p):
        return jax.grad(inner_loss, (0, 1))(p, targets, rollout, env)
    _, (h_params, h_targets) = jax.jvp(grad_both, (params,), (v,))
    return h_params, h_targets, None

inner_grad.defvjp(inner_grad_fwd, inner_grad_bwd)

def inner_step(params, opt_state, eta, rollout, env, mixflow=True):  # targets: constants for the agent, depend on eta
    outputs = unroll(params, rollout, env)
    values = q_stats(outputs, rollout) if USE_Q else None
    raw = rule(eta, *features(outputs, rollout, values), rollout)
    targets = anchor(raw, outputs) + ((values["g"],) if USE_Q else ())
    grads = (inner_grad if mixflow else jax.grad(inner_loss))(params, targets, rollout, env)
    params, opt_state = adam(params, opt_state, grads)
    reg = 1e-3 * sum(t.mean() ** 2 for t in raw) + 1e-2 * kl(stop(outputs["pi"][:-1]), targets[0]).mean()
    return params, opt_state, reg, stop(outputs["pi"][:-1])  # the policy before the update, for the meta-critic

def frozen_step(params, opt_state, eta, rollout, env):  # inner_step's update for a fixed rule (no meta-gradient), with
    # one agent forward pass for the rule's inputs and, by its vjp, the gradient. inner_step runs it twice; XLA merges
    # the two, except for the checkpointed frame stem (ARC: 15% of the time)
    outputs, pull = jax.vjp(lambda p: unroll(p, rollout, env), params)
    values = q_stats(outputs, rollout) if USE_Q else None
    raw = rule(eta, *features(outputs, rollout, values), rollout)
    targets = stop(anchor(raw, outputs) + ((values["g"],) if USE_Q else ()))
    grads = pull(jax.grad(lambda o: outputs_loss(drop_last(o), targets, rollout))(outputs))[0]
    return adam(params, opt_state, grads)

# ---------------------------------------------------------------- replay and one agent update, as in evaluation
def flip(rollout):  # time-major <-> batch-major; h0 has no time axis
    return {k: v if k == "h0" else jnp.swapaxes(v, 0, 1) for k, v in rollout.items()}

def init_buffer(env="catch"):  # BUFFER transitions as trajectories of ROLL steps, batch-major
    n = BUFFER // ROLL
    def zeros(shape, dtype=jnp.float32):
        return jnp.zeros((n,) + shape, dtype)
    s, valid = zeros((ROLL + 1, 3), jnp.int32), dict(valid=zeros((ROLL + 1, A), bool)) if ACT_MAX else {}
    if env in ENVS:  # observations and legal actions
        o = jax.eval_shape(lambda k: ENVS[env].obs(ENVS[env].reset(k, 0, 1)), jax.random.PRNGKey(0))
        s, valid = zeros((ROLL + 1,) + o.shape[1:], o.dtype), dict(valid=zeros((ROLL + 1, A), bool))
    trajectories = dict(s=s, first=zeros((ROLL + 1,)), a=zeros((ROLL,), jnp.int32), r=zeros((ROLL,)),
                        done=zeros((ROLL,)), mu=zeros((ROLL, A)), h0=zeros((memory_size(),))) | valid
    return trajectories | dict(idx=jnp.zeros((), jnp.int32), size=jnp.zeros((), jnp.int32))

def store(buffer, rollout):  # the B new trajectories overwrite the oldest ones
    n = len(buffer["a"])
    i = (buffer["idx"] + jnp.arange(B)) % n
    new = {k: buffer[k].at[i].set(v) for k, v in flip(rollout).items()}
    return new | dict(idx=(buffer["idx"] + B) % n, size=jnp.minimum(n, buffer["size"] + B))

def sample(buffer, rollout, key):  # BATCH trajectories: a REPLAY share from the buffer, the rest from the new rollout
    if REPLAY == 0 and BATCH == B:  # v2: learn on exactly the new rollout
        return rollout
    key_old, key_new, key_mix = jax.random.split(key, 3)
    new = flip(rollout)
    old_i = jax.random.randint(key_old, (BATCH,), 0, buffer["size"])
    new_i = jax.random.randint(key_new, (BATCH,), 0, B)
    replayed = jax.random.bernoulli(key_mix, REPLAY, (BATCH,))
    def mix(k):
        old = buffer[k][old_i]
        return jnp.where(replayed.reshape((-1,) + (1,) * (old.ndim - 1)), old, new[k][new_i])
    return flip({k: mix(k) for k in new})

def update_keys(key):  # one key per update on a rollout; with UTD 1 the key itself, so old results stay the same
    return jnp.stack([key] + [jax.random.fold_in(key, i) for i in range(1, UTD)])

def agent_update(eta, agent, key, env="catch", mixflow=True, fit=True):  # collect, store, UTD x (sample, learn);
    # mixflow True: MixFlow meta-gradient, False: plain, None: frozen rule (frozen_step, no meta-critic)
    params, opt_state, critic, critic_opt_state, carry, buffer = agent
    key_collect, key_sample = jax.random.split(key)
    if ACT_MAX and ACT_REDRAW:  # the agent keeps its weights, but its slots now act differently
        key_slots = jax.random.fold_in(key, 7)
        redraw = jax.random.bernoulli(key_slots, ACT_REDRAW)
        carry = carry[:3] + (jnp.where(redraw, draw_slots(jax.random.fold_in(key_slots, 1), env), carry[3]),)
    carry, rollout = collect(stop(params), carry, key_collect, env)
    buffer = store(buffer, rollout)
    ready = buffer["size"] >= BATCH  # learning starts once the buffer holds one batch, as in evaluation
    def learn(state, key):
        params, opt_state, critic, critic_opt_state = state
        batch = sample(buffer, rollout, key)
        if mixflow is None:  # a frozen rule (no meta-gradient): frozen_step, and fit is False
            (new_params, new_opt_state), reg = frozen_step(params, opt_state, eta, batch, env), jnp.zeros(())
        else:
            step = jax.checkpoint(inner_step, static_argnums=(4, 5)) if REMAT else inner_step
            new_params, new_opt_state, reg, pi = step(params, opt_state, eta, batch, env, mixflow)
        def keep_ready(new, old):
            return jnp.where(ready, new, old)
        params, opt_state = jax.tree.map(keep_ready, (new_params, new_opt_state), (params, opt_state))
        if fit:  # the meta-critic learns V-trace values of the agent's current policy
            critic, critic_opt_state = fit_critic(critic, critic_opt_state, batch, pi)
        return (params, opt_state, critic, critic_opt_state), reg
    state = (params, opt_state, critic, critic_opt_state)
    (params, opt_state, critic, critic_opt_state), reg = jax.lax.scan(learn, state, update_keys(key_sample))
    returns = jnp.stack([rollout["r"].sum(), rollout["done"].sum()])  # for the metric: reward, finished episodes
    return (params, opt_state, critic, critic_opt_state, carry, buffer), (ready * reg.mean(), returns)

# ---------------------------------------------------------------- meta-critic and meta-objective
def init_critic(key):
    keys = jax.random.split(key, 3)
    return dict(l1=dense(keys[0], OBS, HID), l2=dense(keys[1], HID, HID), v=dense(keys[2], HID, 1, 0.01))

def critic_values(critic, rollout):  # on all T+1 states; never sees eta
    x = jax.nn.relu(apply(critic["l2"], jax.nn.relu(apply(critic["l1"], embed(critic, rollout["s"], ENV)))))
    return apply(critic["v"], x)[..., 0]

def critic_targets(critic, rollout, logits):  # stopped V-trace targets and advantages for the policy with these logits
    v = stop(critic_values(critic, rollout))
    return vtrace(v, rollout["r"], GAMMA * (1 - rollout["done"]), importance(logits, rollout), 0.96)

critic_optimizer = optax.adam(CRITIC_LR)
def meta_optimizer():  # built when train runs, so META_LR and META_SHARED set after import take effect
    adam = optax.adam(META_LR)
    return adam if META_SHARED else optax.chain(optax.clip_by_global_norm(1.0), adam)  # one per agent unless shared

def fit_critic(critic, critic_opt_state, rollout, logits):  # loss 0.5 * mean((v - stopped V-trace targets)^2)
    v, pull = jax.vjp(lambda c: critic_values(c, rollout), critic)  # one forward pass for the targets and the gradient
    targets = vtrace(stop(v), rollout["r"], GAMMA * (1 - rollout["done"]), importance(logits, rollout), 0.96)[0]
    grads = pull(jnp.concatenate([v[:-1] - targets, jnp.zeros_like(v[-1:])]) / targets.size)[0]
    updates, critic_opt_state = critic_optimizer.update(grads, critic_opt_state)
    return optax.apply_updates(critic, updates), critic_opt_state

def merge(rollouts):  # n stacked rollouts [n, T, B, ...] -> one rollout of n * B trajectories
    def join(k, x):
        if k == "h0":
            return x.reshape((-1,) + x.shape[2:])
        return jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1, *x.shape[3:])
    return {k: join(k, x) for k, x in rollouts.items()}

def meta_loss(eta, agent, key, mixflow=True):  # INNER agent updates, then the adapted agent collects data and is scored
    key_train, key_valid = jax.random.split(key)
    def update(agent, key):
        return agent_update(eta, agent, key, ENV, mixflow)
    agent, (regs, returns) = jax.lax.scan(update, agent, jax.random.split(key_train, INNER))
    params, critic, carry = agent[0], agent[2], agent[4]
    def one(carry, key):
        return collect(stop(params), carry, key, ENV)
    carry, valid = jax.lax.scan(one, carry, jax.random.split(key_valid, VALID))
    valid = merge(valid)
    outputs = drop_last(unroll(params, valid, ENV))
    advantages = critic_targets(critic, valid, outputs["pi"])[1]
    logp = pick(lsm(outputs["pi"]), valid["a"])[..., 0]
    loss = -(logp * stop(norm(advantages))).mean() - META_ENT * entropy(outputs["pi"])
    loss -= 1e-3 * (entropy(outputs["y"]) + entropy(take(outputs["z"], valid["a"])))
    returns = returns.sum(0) + jnp.stack([valid["r"].sum(), valid["done"].sum()])
    agent = agent[:4] + (carry,) + agent[5:]
    return loss + regs.mean(), (agent, returns[0] / jnp.maximum(1.0, returns[1]))

# ---------------------------------------------------------------- one jit for all of meta-training
def init_agent_state(key, env=None, game=0):  # agent, Adam state, meta-critic, critic optimizer, env carry, buffer
    env = env or ENV
    key_agent, key_critic, key_env = jax.random.split(key, 3)
    params = with_stem(init_agent(key_agent), jax.random.fold_in(key_agent, 1), env)
    critic = with_stem(init_critic(key_critic), jax.random.fold_in(key_critic, 1), env)
    zeros = jax.tree.map(jnp.zeros_like, params)
    carry = (env_reset(key_env, env, game), jnp.zeros((B, memory_size())), jnp.ones(B))
    carry += (draw_slots(jax.random.fold_in(key_env, 1), env),) if ACT_MAX else ()
    return params, (zeros, zeros, jnp.zeros(())), critic, critic_optimizer.init(critic), carry, init_buffer(env)

def init_agents(key, n, env, draw=False):  # n fresh agents: agent i on game i mod the number of games, or random games
    games = agent_games(n, env, jax.random.fold_in(key, 1) if draw else None)
    return jax.vmap(lambda k, g: init_agent_state(k, env, g))(jax.random.split(key, n), games)

def draw_lifetimes(key, lifetimes, n):  # probability proportional to 1/lifetime: equal experience per category
    lifetimes = jnp.asarray(lifetimes)
    return jax.random.choice(key, lifetimes, (n,), p=(1 / lifetimes) / (1 / lifetimes).sum())

def init_train(key, lifetimes=LIFETIMES):  # meta-training state: rule, meta-optimizer states, agents, ages, lifetimes
    opt, (key_eta, key_agents, _) = meta_optimizer(), jax.random.split(key, 3)
    eta, agents = init_rule(key_eta), init_agents(key_agents, AGENTS, ENV)
    meta_opt_state = opt.init(eta) if META_SHARED else jax.vmap(lambda _: opt.init(eta))(jnp.arange(AGENTS))
    age = jnp.zeros(AGENTS, jnp.int32)
    lifetime = draw_lifetimes(jax.random.fold_in(key_agents, 1), lifetimes, AGENTS) if lifetimes else age
    return eta, meta_opt_state, agents, age, lifetime

def train(key, steps, meta_lr_scale=1.0, mixflow=True, lifetimes=LIFETIMES, state=None):  # meta_lr_scale 0: frozen
    # rule; state: go on from init_train or an earlier train (None: init_train(key)). Returns the rule, state, logs
    opt = meta_optimizer()
    def one_agent(eta, agent, key):
        (loss, (agent, ret)), grads = jax.value_and_grad(meta_loss, has_aux=True)(eta, agent, key, mixflow)
        return grads, agent, (loss, ret)
    def meta_step(state, key):
        eta, meta_opt_state, agents, age, lifetime = state
        key_reset, key_lifetime, key = jax.random.split(key, 3)
        if lifetimes:  # replace agents whose lifetime is over: new agent, Adam state, critic, buffer, and a random game
            due, fresh = age >= lifetime, init_agents(key_reset, AGENTS, ENV, draw=True)
            def select(new, old):
                return jnp.where(due.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)
            agents = jax.tree.map(select, fresh, agents)
            age = jnp.where(due, 0, age)
            lifetime = jnp.where(due, draw_lifetimes(key_lifetime, lifetimes, AGENTS), lifetime)
        grads, agents, (loss, ret) = jax.vmap(one_agent, (None, 0, 0))(eta, agents, jax.random.split(key, AGENTS))
        if META_SHARED:
            if META_SHARED == "clip":
                scale = 1 / jnp.maximum(jax.vmap(optax.global_norm)(grads), 1.0)
                grads = jax.tree.map(lambda g: g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), grads)
            updates, meta_opt_state = opt.update(jax.tree.map(lambda g: g.mean(0), grads), meta_opt_state, eta)
        else:
            updates, meta_opt_state = jax.vmap(opt.update, (0, 0, None))(grads, meta_opt_state, eta)
            updates = jax.tree.map(lambda u: u.mean(0), updates)  # mean of the agents' Adam steps
        updates = jax.tree.map(lambda u: meta_lr_scale * u, updates)
        logs = dict(meta_loss=loss.mean(), train_return=ret.mean(), mean_age=age.mean())
        return (optax.apply_updates(eta, updates), meta_opt_state, agents, age + 1, lifetime), logs
    state = init_train(key, lifetimes) if state is None else state
    state, logs = jax.lax.scan(meta_step, state, jax.random.split(jax.random.split(key, 3)[2], steps))
    return state[0], state, logs

def run_frozen(eta, key, steps, env="catch"):  # fresh agents, `steps` updates by a fixed rule; return per update
    def one_agent(agent, key):
        agent, (_, returns) = agent_update(eta, agent, key, env, None, fit=False)
        return agent, returns
    def step(agents, key):
        agents, returns = jax.vmap(one_agent)(agents, jax.random.split(key, AGENTS))
        return agents, returns.sum(0)
    key_agents, key_steps = jax.random.split(key)
    agents = init_agents(key_agents, AGENTS, env)
    returns = jax.lax.scan(step, agents, jax.random.split(key_steps, steps))[1]
    return returns[:, 0] / jnp.maximum(1.0, returns[:, 1])

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    eta, _, logs = jax.jit(lambda key: train(key, steps))(jax.random.PRNGKey(0))
    r, every = logs["train_return"], max(1, steps // 20)
    for i in range(0, steps, every):
        print(f"meta-step {i:5d}  training return {float(r[i : i + every].mean()):+.3f}")
