"""Gymnasium environment over the HKRLBot mod protocol (v2)."""
import sys
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hkrl.bosses import get_boss
from hkrl.protocol import Connection, ConnectionClosed, PROTOCOL_VERSION
from hkrl.reset_metrics import append_reset_span, reset_log_path

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
    "health_bonus": 1.0,     # per mask remaining, on a win
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
    # `reset_log_dir`, when set, turns on Phase 0 sibling-freeze measurement:
    # every reset() appends its wall-clock span to a per-port sidecar under
    # that directory (see hkrl/reset_metrics.py). Off by default -- N=1 runs,
    # tests, and non-measurement training pay nothing.
    def __init__(self, host="127.0.0.1", port=9020, reward_config=None,
                 max_steps=2700, timeout=30.0, reset_retries=8,
                 keepalive=3.0, reset_log_dir=None, boss="hornet1"):
        # Resolved before any socket work so an unknown id fails instantly
        # and locally, not after a game connection is already up.
        self.boss = get_boss(boss)
        self.reward = dict(DEFAULT_REWARD, **(reward_config or {}))
        self.max_steps = max_steps
        self.reset_retries = reset_retries
        self._reset_log = (reset_log_path(reset_log_dir, port)
                           if reset_log_dir is not None else None)
        self.conn = Connection(host=host, port=port, timeout=timeout,
                               keepalive=keepalive)
        self.conn.connect()
        version = (self.conn.hello or {}).get("version")
        if version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"mod speaks protocol v{version}, this trainer needs "
                f"v{PROTOCOL_VERSION} -- rebuild the mod (mod/build.sh) and "
                f"restart the game")
        self._reset_abort = threading.Event()
        self._steps = 0
        self._prev = None
        self._max_bhp = None
        self._warned_states = set()
        n = len(OBS_KEYS) + len(self.boss.fsm_states)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

    # -- helpers --

    def _flatten(self, obs):
        if self._max_bhp is None or obs["bhp"] > self._max_bhp:
            self._max_bhp = max(obs["bhp"], 1)
        b = self.boss
        v = [
            (obs["kx"] - b.arena_center_x) / b.arena_half_w,
            (obs["ky"] - b.floor_y) / b.arena_height,
            obs["kvx"] / VEL_SCALE, obs["kvy"] / VEL_SCALE,
            obs["khp"] / 9.0, obs["soul"] / 99.0,
            float(obs["on_ground"]), float(obs["dashing"]),
            float(obs["invuln"]), float(obs["facing_right"]),
            (obs["bx"] - obs["kx"]) / b.arena_half_w,
            (obs["by"] - obs["ky"]) / b.arena_height,
            obs["bvx"] / VEL_SCALE, obs["bvy"] / VEL_SCALE,
            obs["bhp"] / self._max_bhp,
            float(obs["needle_active"]),
            (obs["nx"] - obs["kx"]) / b.arena_half_w if obs["needle_active"] else 0.0,
            (obs["ny"] - obs["ky"]) / b.arena_height if obs["needle_active"] else 0.0,
        ]
        onehot = [0.0] * len(b.fsm_states)
        try:
            onehot[b.fsm_states.index(obs["boss_state"])] = 1.0
        except ValueError:
            onehot[-1] = 1.0
            unseen = obs["boss_state"]
            if unseen and unseen not in self._warned_states:
                self._warned_states.add(unseen)
                print(f"hkrl: boss_state {unseen!r} is not in {self.boss.id}'s "
                      f"registry list; mapped to UNKNOWN. If this repeats, it "
                      f"is a candidate for the fsm_states list.",
                      file=sys.stderr, flush=True)
        return np.asarray(v + onehot, dtype=np.float32)

    def _reward(self, prev, cur, done, won, truncated):
        r = self.reward["time_penalty"]
        if cur["bhp"] < prev["bhp"]:
            r += (prev["bhp"] - cur["bhp"]) * self.reward["boss_hp_scale"]
        if cur["khp"] < prev["khp"]:
            r += (prev["khp"] - cur["khp"]) * self.reward["knight_hit"]
        if done:
            if won:
                r += self.reward["win"] + cur["khp"] * self.reward["health_bonus"]
            else:
                r += self.reward["death"]
        elif truncated:
            # Running out the clock is a loss, not a free exit: without this
            # a timeout undercuts dying and stalling becomes the best play.
            r += self.reward["death"]
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
        # Span the whole attempt, retries and reconnects included: that full
        # duration is exactly how long step_wait() blocks the fleet on this
        # instance -- the sibling-freeze cost Phase 0 measures.
        started = time.perf_counter()
        for retry in range(self.reset_retries + 1):
            try:
                self.conn.send({"type": "reset", "boss": self.boss.id})
                msg = self.conn.recv()
                if msg.get("type") == "error":
                    # The mod refused the reset (e.g. a boss id its registry
                    # doesn't know). A registry-skew bug, not protocol
                    # rhythm: fail loudly instead of retrying into the same
                    # refusal.
                    raise RuntimeError(
                        f"mod refused reset: {msg.get('message', msg)}")
                break
            except (ConnectionClosed, BrokenPipeError, ConnectionResetError):
                if self._reset_abort.is_set():
                    raise
                if retry == self.reset_retries:
                    raise
                # stderr like the supervisor's lines: SB3's tables own stdout.
                print(f"hkrl: mod dropped the connection during reset "
                      f"(retry {retry + 1}/{self.reset_retries}; a reset-budget "
                      f"expiry while the game boots or unwinds is normal) -- "
                      f"reconnecting", file=sys.stderr, flush=True)
                self.conn.close()
                self.conn.connect()
                if self._reset_abort.is_set():
                    # abort_reset() fired while we were reconnecting: its
                    # shutdown hit the old socket, so honor the request here
                    # instead of blindly retrying against the fresh one.
                    self.conn.abort()
                    raise
        if self._reset_log is not None:
            append_reset_span(self._reset_log,
                              span_s=time.perf_counter() - started,
                              t=time.monotonic())
        self._prev = msg["obs"]
        self._steps = 0
        self._max_bhp = None
        return self._flatten(msg["obs"]), dict(msg["info"])

    def abort_reset(self):
        """Abandon an in-flight reset() from another thread (the async-reset
        shutdown path): shut the socket down so a blocked recv returns now
        rather than at the 30s timeout, and make the retry loop re-raise
        instead of treating the drop as the protocol's normal rhythm and
        reconnecting."""
        self._reset_abort.set()
        self.conn.abort()

    def step(self, action):
        self.conn.send({"type": "action", "buttons": ACTIONS[int(action)]})
        msg = self.conn.recv()
        cur, info = msg["obs"], dict(msg["info"])
        done, won = bool(msg["done"]), bool(info.get("won", False))
        truncated = not done and self._steps + 1 >= self.max_steps
        reward = self._reward(self._prev, cur, done, won, truncated)
        self._prev = cur
        self._steps += 1
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
