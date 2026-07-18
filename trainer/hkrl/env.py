"""Gymnasium environment over the HKRLBot mod protocol (v1)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hkrl.protocol import Connection

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
VEL_SCALE = 20.0

OBS_KEYS = [  # scalar block order (before the boss-state one-hot)
    "kx", "ky", "kvx", "kvy", "khp", "soul",
    "on_ground", "dashing", "invuln", "facing_right",
    "bx", "by", "bvx", "bvy", "bhp",
    "needle_active", "nx", "ny",
]

_BUTTONS = ["left", "right", "up", "down", "jump", "attack", "dash"]


def _b(**kw):
    d = {k: False for k in _BUTTONS}
    d.update(kw)
    return d


ACTIONS = [
    _b(),
    _b(left=True), _b(right=True),
    _b(jump=True), _b(left=True, jump=True), _b(right=True, jump=True),
    _b(attack=True), _b(left=True, attack=True), _b(right=True, attack=True),
    _b(dash=True), _b(left=True, dash=True), _b(right=True, dash=True),
    _b(up=True, attack=True), _b(down=True, attack=True),
    _b(jump=True, attack=True),
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

    # Final-review fix (F5): the mod's socket read (BridgeServer.cs, the
    # stream.ReadTimeout set in AcceptLoop) has a hard 10s ceiling: if this
    # process doesn't send its next message ("action" or "reset") within 10s
    # of the mod's last SendState, the mod's blocking ReadLine() times out and
    # it drops the connection outright -- step()/reset() below then observe
    # this as ConnectionClosed, not a retryable timeout. This means any
    # blocking work performed between receiving a state and calling step()/
    # reset() again -- e.g. a PPO policy update running synchronously on this
    # thread between rollout steps -- must complete in well under 10s, or the
    # connection (and the in-progress episode) will be silently killed. Do
    # not change the mod's ReadTimeout to "fix" this here; if a training loop
    # needs longer than 10s between messages, that ceiling has to be revisited
    # on the mod side instead.

    def __init__(self, host="127.0.0.1", port=9020, reward_config=None, max_steps=2700):
        self.reward = dict(DEFAULT_REWARD, **(reward_config or {}))
        self.max_steps = max_steps
        self.conn = Connection(host=host, port=port)
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
            (obs["ky"] - FLOOR_Y) / ARENA_HALF_W,
            obs["kvx"] / VEL_SCALE, obs["kvy"] / VEL_SCALE,
            obs["khp"] / 9.0, obs["soul"] / 99.0,
            float(obs["on_ground"]), float(obs["dashing"]),
            float(obs["invuln"]), float(obs["facing_right"]),
            (obs["bx"] - obs["kx"]) / ARENA_HALF_W,
            (obs["by"] - obs["ky"]) / ARENA_HALF_W,
            obs["bvx"] / VEL_SCALE, obs["bvy"] / VEL_SCALE,
            obs["bhp"] / self._max_bhp,
            float(obs["needle_active"]),
            (obs["nx"] - obs["kx"]) / ARENA_HALF_W if obs["needle_active"] else 0.0,
            (obs["ny"] - obs["ky"]) / ARENA_HALF_W if obs["needle_active"] else 0.0,
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.conn.send({"type": "reset"})
        msg = self.conn.recv()
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
        return self._flatten(cur), reward, done, truncated, info

    def close(self):
        self.conn.close()
