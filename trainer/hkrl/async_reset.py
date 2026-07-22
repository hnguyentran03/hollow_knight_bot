"""Async episode resets: keep siblings stepping while one instance resets.

SB3's SubprocVecEnv worker auto-resets a done env synchronously inside
step(), so one instance's multi-second reset macro freezes every sibling in
the lockstep step_wait (measured on a real N=2 run -- see
docs/superpowers/specs/2026-07-21-async-resets-design.md, Phase 0). This
wrapper moves the reset to a background thread inside the worker and emits
placeholder transitions until it completes, so siblings keep taking real
steps.

Recv ownership: while a reset is pending, the background thread is the sole
user of the inner env (and thus of its connection's single-reader recv
slot). The wrapper never touches the inner env until the thread is joined.
Actions are swallowed, not sent: the mod reads nothing off its socket while
awaitingReset, so a sent action would queue in the TCP buffer and desync
the lockstep protocol when the fight goes live.

The placeholder observation is all zeros in the UNCHANGED observation
space. For HKEnv that is deliberately out-of-distribution without a new
feature: every real observation carries exactly one hot bit in the
boss-state block, so all-zeros is unreachable -- and the 46-dim space is
preserved, keeping --resume and every existing checkpoint working.

Two pending-window semantics, compared by the Phase 2 gate:
- "prefix": placeholders open the next episode and the fresh observation is
  spliced in mid-episode (done=False), so LSTM hidden state carries across
  the pending->live boundary.
- "isolated": the pending window is its own throwaway episode, ended
  (terminated=True, all rewards 0) once the reset completes; the fresh
  observation is delivered by the auto-reset that follows, so the real
  episode contains no placeholders and LSTM state resets at the boundary.
"""
import threading
import time

import gymnasium as gym
import numpy as np

PENDING_MODES = ("prefix", "isolated")


class AsyncResetWrapper(gym.Wrapper):
    """Non-blocking auto-resets for an env whose reset() takes seconds.

    Only the auto-reset after a finished episode is asynchronous: the
    initial reset -- and any explicit reset when no episode just finished,
    e.g. after a supervisor rebuild -- stays synchronous, because training
    cannot start on placeholders and a rebuilt vec must come up with no
    inherited pending state. `placeholder_tick_s` paces placeholder steps
    so an all-pending batch cannot spin at CPU speed.
    """

    def __init__(self, env, pending_mode="prefix", placeholder_tick_s=0.067,
                 close_join_s=5.0):
        super().__init__(env)
        if pending_mode not in PENDING_MODES:
            raise ValueError(f"pending_mode must be one of {PENDING_MODES}")
        self.pending_mode = pending_mode
        self.placeholder_tick_s = placeholder_tick_s
        self.close_join_s = close_join_s
        self._placeholder = np.zeros(env.observation_space.shape,
                                     dtype=env.observation_space.dtype)
        self._thread = None   # sole owner of the inner env while alive
        self._result = None   # (obs, info) from a completed background reset
        self._error = None    # exception from a failed background reset
        self._fresh = None    # completed reset awaiting delivery ("isolated")
        self._episode_over = False  # last real step ended its episode

    def reset(self, seed=None, options=None):
        if self._fresh is not None:
            # "isolated": the throwaway placeholder episode just ended; this
            # auto-reset delivers the fresh observation the background
            # thread already produced.
            obs, info = self._fresh
            self._fresh = None
            return obs, info
        if self._thread is not None:
            # Explicit reset mid-pending (e.g. learn() start): complete the
            # recv handoff, then fall through to a genuinely fresh
            # synchronous reset -- the joined result may be seconds stale.
            self._join_pending()
        if self._episode_over:
            self._episode_over = False
            self._begin(seed, options)
            return self._placeholder.copy(), {"reset_pending": True}
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._thread is not None:
            if self._thread.is_alive():
                # Swallow the action (see module docstring) and pace the
                # tick so all-pending batches don't spin at CPU speed.
                time.sleep(self.placeholder_tick_s)
                return (self._placeholder.copy(), 0.0, False, False,
                        {"reset_pending": True})
            obs, info = self._join_pending()
            if self.pending_mode == "isolated":
                self._fresh = (obs, info)
                return (self._placeholder.copy(), 0.0, True, False,
                        {"reset_pending": True})
            # "prefix": splice the fresh observation into the running
            # episode. This action is swallowed too -- it was chosen
            # against a placeholder, not against this observation.
            return obs, 0.0, False, False, dict(info, reset_pending=False)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_over = terminated or truncated
        return obs, reward, terminated, truncated, info

    def close(self):
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Unblock a reset thread parked in a blocking recv (up to the
            # 30s socket timeout) rather than waiting it out. Without an
            # abort_reset on the inner env, the bounded join abandons the
            # daemon thread -- acceptable only because the worker process
            # exits right after close.
            abort = getattr(self.env, "abort_reset", None)
            if abort is not None:
                abort()
            thread.join(timeout=self.close_join_s)
        self._thread = None
        self._error = None
        self.env.close()

    def _begin(self, seed, options):
        self._result = None
        self._error = None

        def run():
            try:
                self._result = self.env.reset(seed=seed, options=options)
            except BaseException as error:  # noqa: BLE001
                # A raise in a thread kills nothing. Captured here and
                # re-raised from _join_pending on the worker's next call, so
                # the worker process dies and the supervisor's EOFError
                # recovery path engages (design doc section 6).
                self._error = error

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _join_pending(self):
        self._thread.join()
        self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        result, self._result = self._result, None
        return result
