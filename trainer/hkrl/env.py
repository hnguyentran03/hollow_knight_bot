"""Gymnasium environment over the HKRLBot mod protocol (v1)."""
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hkrl.protocol import Connection, ConnectionClosed

# Hornet 1 FSM states, recorded from live play in Hall of Gods (Hornet 1,
# Attuned), Godhome. See mod/DISCOVERED.md section 1 ("Hornet FSM state
# names") for the source measurement. Unknown/unseen states map to the
# trailing "UNKNOWN" fallback slot.
HORNET_STATES = [
    "Flourish", "Run", "A Dash", "Hard Land", "Idle", "Throw Antic",
    "Thrown", "Throw Recover", "In Air", "ADash Antic", "Run Antic",
    "G Dash Antic", "G Dash", "Jump Antic", "Land", "GDash Recover1",
    "GDash Recover2", "Evade", "Evade Antic", "Evade Land", "Wall L",
    "Sphere A", "Sphere Antic A", "Sphere Recover A", "Wall R",
    "Stun Air", "Stun Land",
    "UNKNOWN",
]

# Normalization constants. Recorded from live play in Hall of Gods (Hornet 1
# arena), see mod/DISCOVERED.md section 2 ("Arena bounds").
#   Knight X at left wall:  15.27
#   Knight X at right wall: 37.73
#   Floor Y:                28.41
#   Arena center X = (15.27 + 37.73) / 2 = 26.5
#   Arena half-width X = (37.73 - 15.27) / 2 = 11.23
ARENA_CENTER_X = 26.5
ARENA_HALF_W = 11.23
FLOOR_Y = 28.41
# Vertical scale for all vertical position terms: the arena's full height above
# the floor (top - floor = 38 - 28.41), measured off the F1 overlay per
# DISCOVERED.md section 2. NOT a "half" like ARENA_HALF_W -- vertical is
# normalized floor-relative ((ky - FLOOR_Y) / ARENA_HEIGHT: floor -> 0, top ->
# ~1), not center-relative, so there is no /2. The horizontal half-width MUST
# NOT be reused here: the arena is far wider than it is tall, so normalizing a
# vertical offset by 11.23 would squash every jump/leap into a sliver near zero
# and make vertical spacing nearly invisible to the policy.
ARENA_HEIGHT = 9.59
VEL_SCALE = 20.0

OBS_KEYS = [  # scalar block order (before the boss-state one-hot)
    "kx", "ky", "kvx", "kvy", "khp", "soul",
    "on_ground", "dashing", "invuln", "facing_right",
    "bx", "by", "bvx", "bvy", "bhp",
    "needle_active", "nx", "ny",
]

_BUTTONS = ["left", "right", "up", "down", "jump", "attack", "dash",
            "cast", "focus"]


def _b(**kw):
    d = {k: False for k in _BUTTONS}
    d.update(kw)
    return d


# Each entry is one executable move, not a raw button: the directional
# attack/cast variants exist so the Knight can keep moving while it swings
# or casts, and up/down variants are distinct moves in their own right
# (up-slash, pogo, Wraiths, Dive). Temporally extended moves are not listed
# separately -- buttons stay held for the whole 67ms tick and across
# consecutive steps that repeat them, so jump height comes from how many
# steps in a row hold jump, and a heal is `focus` held ~20 consecutive
# steps standing still. `cast` maps to the game's Quick Cast (spell fires
# on press, direction picks Vengeful Spirit / Wraiths / Dive); `focus`
# maps to Focus/Cast, whose hold is what channels healing.
ACTIONS = [
    _b(),
    _b(left=True), _b(right=True),
    _b(jump=True), _b(left=True, jump=True), _b(right=True, jump=True),
    _b(attack=True), _b(left=True, attack=True), _b(right=True, attack=True),
    _b(dash=True), _b(left=True, dash=True), _b(right=True, dash=True),
    _b(up=True, attack=True), _b(down=True, attack=True),
    _b(jump=True, attack=True),
    _b(cast=True), _b(left=True, cast=True), _b(right=True, cast=True),
    _b(up=True, cast=True), _b(down=True, cast=True),
    _b(focus=True),
]

DEFAULT_REWARD = {
    "boss_hp_scale": 0.03,   # per boss HP point removed
    "knight_hit": -1.0,      # per mask lost
    "win": 10.0,
    "death": -5.0,
    "time_penalty": -0.001,  # per decision step
}


class HKEnv(gym.Env):
    metadata = {"render_modes": []}

    # The mod's socket read (BridgeServer.cs, the stream.ReadTimeout set in
    # AcceptLoop) has a hard 10s ceiling per read. The Connection's keepalive
    # pinger (hkrl/protocol.py) keeps this connection under that ceiling
    # through any lockstep gap -- another slot's multi-second episode reset,
    # a slow PPO update -- so idle time no longer silently kills the
    # connection and the in-progress episode. What the pinger does NOT
    # change: the game keeps running in real time while the trainer thinks,
    # so long think-time still means the Knight stands in a live boss fight
    # eating hits. Keep updates short (see the --n-epochs note in
    # scripts/train.py) for the fight's sake, not the socket's.

    # `timeout` is the socket read deadline for every message, including the
    # `hello` read inside connect(). It bounds how long a wedged instance can
    # stall this env before the failure becomes visible to the supervisor.
    # `reset_retries` bounds reset()'s reconnect-and-retry loop (see reset()
    # below). Sized like the supervisor's recover_attempts=8 in
    # scripts/train.py and for the same reason: a cold boot-to-fight spans
    # several of the mod's 22.5s reset budgets, and each expiry costs one
    # retry here.
    def __init__(self, host="127.0.0.1", port=9020, reward_config=None,
                 max_steps=2700, timeout=30.0, reset_retries=8,
                 keepalive=3.0):
        self.reward = dict(DEFAULT_REWARD, **(reward_config or {}))
        self.max_steps = max_steps
        self.reset_retries = reset_retries
        self.conn = Connection(host=host, port=port, timeout=timeout,
                               keepalive=keepalive)
        self.conn.connect()
        self._steps = 0
        self._prev = None
        self._max_bhp = None
        n = len(OBS_KEYS) + len(HORNET_STATES)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

    # -- helpers --

    def _flatten(self, obs):
        if self._max_bhp is None or obs["bhp"] > self._max_bhp:
            self._max_bhp = max(obs["bhp"], 1)
        v = [
            (obs["kx"] - ARENA_CENTER_X) / ARENA_HALF_W,
            (obs["ky"] - FLOOR_Y) / ARENA_HEIGHT,
            obs["kvx"] / VEL_SCALE, obs["kvy"] / VEL_SCALE,
            obs["khp"] / 9.0, obs["soul"] / 99.0,
            float(obs["on_ground"]), float(obs["dashing"]),
            float(obs["invuln"]), float(obs["facing_right"]),
            (obs["bx"] - obs["kx"]) / ARENA_HALF_W,
            (obs["by"] - obs["ky"]) / ARENA_HEIGHT,
            obs["bvx"] / VEL_SCALE, obs["bvy"] / VEL_SCALE,
            obs["bhp"] / self._max_bhp,
            float(obs["needle_active"]),
            (obs["nx"] - obs["kx"]) / ARENA_HALF_W if obs["needle_active"] else 0.0,
            (obs["ny"] - obs["ky"]) / ARENA_HEIGHT if obs["needle_active"] else 0.0,
        ]
        onehot = [0.0] * len(HORNET_STATES)
        try:
            onehot[HORNET_STATES.index(obs["boss_state"])] = 1.0
        except ValueError:
            onehot[-1] = 1.0
        return np.asarray(v + onehot, dtype=np.float32)

    def _reward(self, prev, cur, done, won):
        r = self.reward["time_penalty"]
        if cur["bhp"] < prev["bhp"]:
            r += (prev["bhp"] - cur["bhp"]) * self.reward["boss_hp_scale"]
        if cur["khp"] < prev["khp"]:
            r += (prev["khp"] - cur["khp"]) * self.reward["knight_hit"]
        if done:
            r += self.reward["win"] if won else self.reward["death"]
        return r

    # -- gym API --

    # A reset the mod drops is retried by reconnecting, because those drops
    # are part of the protocol's normal rhythm, not failures: the mod's
    # reset macro has a 22.5s budget (deliberately under this socket's 30s
    # timeout), a cold boot-to-fight -- title menu, standing up from the
    # Hall of Gods bench, the walk to the statue, the challenge menu -- runs
    # ~25s+, and menu/scene progress persists across drops, so successive
    # resets ratchet forward until the fight is live. Handling the drop here
    # keeps it invisible to the layers above; escaping instead would kill
    # this env's vec worker and turn every boot budget expiry into a full
    # supervisor recovery (observed live: a healthy mid-boot game "recovered"
    # with a vec rebuild at t=0 of every cold-started run).
    #
    # Only the drop is retried. A socket timeout (wedged main thread) and a
    # refused reconnect (dead game) still propagate: those are exactly the
    # cases the supervisor's relaunch machinery exists for, and retrying
    # them here would only delay it.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for retry in range(self.reset_retries + 1):
            try:
                self.conn.send({"type": "reset"})
                msg = self.conn.recv()
                break
            except (ConnectionClosed, BrokenPipeError, ConnectionResetError):
                if retry == self.reset_retries:
                    raise
                # stderr like the supervisor's lines: SB3's tables own stdout.
                print(f"hkrl: mod dropped the connection during reset "
                      f"(retry {retry + 1}/{self.reset_retries}; a reset-budget "
                      f"expiry while the game boots or unwinds is normal) -- "
                      f"reconnecting", file=sys.stderr, flush=True)
                self.conn.close()
                self.conn.connect()
        self._prev = msg["obs"]
        self._steps = 0
        self._max_bhp = None
        return self._flatten(msg["obs"]), dict(msg["info"])

    def step(self, action):
        self.conn.send({"type": "action", "buttons": ACTIONS[int(action)]})
        msg = self.conn.recv()
        cur, info = msg["obs"], dict(msg["info"])
        done, won = bool(msg["done"]), bool(info.get("won", False))
        reward = self._reward(self._prev, cur, done, won)
        self._prev = cur
        self._steps += 1
        truncated = not done and self._steps >= self.max_steps
        if done or truncated:
            # Fraction of the boss's starting HP removed this episode. Lives
            # in terminal info so the random-agent exploration gate and the
            # per-generation training manifest read the same measurement.
            # _max_bhp was pinned by reset()'s _flatten to the first frame's
            # reading (the fight's starting HP); taking the max with the
            # current frame keeps the fraction non-negative if a frame ever
            # reports more than that baseline.
            max_bhp = max(self._max_bhp or 1, cur["bhp"], 1)
            info["boss_damage_frac"] = (max_bhp - cur["bhp"]) / max_bhp
        return self._flatten(cur), reward, done, truncated, info

    def close(self):
        self.conn.close()
