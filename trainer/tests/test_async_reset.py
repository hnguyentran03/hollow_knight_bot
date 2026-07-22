import functools
import time

import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from hkrl.async_reset import AsyncResetWrapper
from hkrl.fake_slow_env import SlowResetEnv, make_async_timed
from hkrl.protocol import ConnectionClosed


def _step_until_live(env, action=0, tries=200):
    """Step through the pending window until the splice, returning the first
    non-placeholder result."""
    for _ in range(tries):
        obs, reward, terminated, truncated, info = env.step(action)
        if not info.get("reset_pending"):
            return obs, reward, terminated, truncated, info
        time.sleep(0.01)
    raise AssertionError("reset never completed")


def test_initial_reset_is_synchronous_and_unmarked():
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    obs, info = env.reset()
    assert (obs == 1).all()
    assert "reset_pending" not in info
    env.close()


def test_auto_reset_after_done_returns_placeholders_and_sends_no_actions():
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    env.reset()
    inner.next_done = True
    _, _, terminated, _, _ = env.step(0)
    assert terminated
    inner.gate.clear()               # the next reset blocks until released
    obs, info = env.reset()          # SB3 worker's auto-reset call
    assert info == {"reset_pending": True}
    assert (obs == 0).all()          # all-zeros placeholder
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(1)
        assert info["reset_pending"] and reward == 0.0
        assert not terminated and not truncated
        assert (obs == 0).all()
    # Only the pre-death action ever reached the game: pending actions are
    # swallowed, not queued (a queued action would desync the protocol).
    assert inner.actions == [0]
    inner.gate.set()
    env.close()


def test_prefix_splice_delivers_the_fresh_obs_without_ending_the_episode():
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    env.reset()
    inner.next_done = True
    env.step(0)
    inner.gate.clear()
    env.reset()
    env.step(1)                      # at least one placeholder tick
    inner.gate.set()                 # background reset can now finish
    obs, reward, terminated, truncated, info = _step_until_live(env, action=1)
    assert (obs == 2).all()          # reset #2's own first frame, spliced in
    assert reward == 0.0 and not terminated and not truncated
    assert info["reset_pending"] is False
    assert info["reset"] == 2        # the inner reset's info rides along
    # The splice consumed no action either (it was chosen against a
    # placeholder); the next step is the first real one.
    assert inner.actions == [0]
    env.step(1)
    assert inner.actions == [0, 1]
    env.close()


def test_placeholder_steps_are_paced_to_the_decision_tick():
    """An all-pending batch would otherwise spin at CPU speed and flood the
    rollout buffer with thousands of placeholder transitions per second."""
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.05)
    env.reset()
    inner.next_done = True
    env.step(0)
    inner.gate.clear()
    env.reset()
    started = time.monotonic()
    for _ in range(3):
        env.step(0)
    assert time.monotonic() - started >= 0.15
    inner.gate.set()
    env.close()


def test_rejects_an_unknown_pending_mode():
    with pytest.raises(ValueError):
        AsyncResetWrapper(SlowResetEnv(), pending_mode="bogus")


def test_isolated_mode_ends_the_pending_window_as_a_throwaway_episode():
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, pending_mode="isolated",
                            placeholder_tick_s=0.0)
    env.reset()
    inner.next_done = True
    env.step(0)
    inner.gate.clear()
    obs, info = env.reset()              # placeholder episode begins
    assert info["reset_pending"] and (obs == 0).all()
    env.step(0)                          # a placeholder tick inside it
    inner.gate.set()
    terminated = False
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break
        time.sleep(0.01)
    # The throwaway episode ends ON a placeholder, so the real episode
    # about to start contains none -- LSTM state resets at the boundary.
    assert terminated and not truncated
    assert (obs == 0).all() and reward == 0.0 and info["reset_pending"]
    obs, info = env.reset()              # auto-reset delivers the fresh frame
    assert (obs == 2).all() and info == {"reset": 2}
    env.step(1)                          # real stepping resumed
    assert inner.actions == [0, 1]
    env.close()


def test_a_background_reset_that_raises_kills_the_next_step():
    """An exception in a thread kills nothing by itself. It must re-raise on
    the worker's next step so the worker process dies and the supervisor's
    EOFError recovery path engages -- otherwise a genuinely dead game means
    placeholders forever (design doc section 6)."""
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    env.reset()
    inner.next_done = True
    env.step(0)
    inner.reset_error = ConnectionClosed("game died mid-reset")
    env.reset()                          # background reset raises immediately
    with pytest.raises(ConnectionClosed, match="mid-reset"):
        for _ in range(200):
            env.step(0)
            time.sleep(0.01)
    env.close()


def test_close_during_a_pending_reset_aborts_instead_of_waiting():
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    env.reset()
    inner.next_done = True
    env.step(0)
    inner.gate.clear()                   # reset thread parks on the gate
    env.reset()
    started = time.monotonic()
    env.close()                          # must abort_reset(), not wait
    assert time.monotonic() - started < 2.0
    assert inner.aborted.is_set()


def test_explicit_reset_while_pending_yields_a_genuinely_fresh_reset():
    """SB3 calls vec.reset() at learn() start. If a background reset is in
    flight the wrapper must complete the recv handoff and then reset again
    for real -- never hand a possibly-stale joined result to a fresh
    rollout, and never a placeholder."""
    inner = SlowResetEnv()
    env = AsyncResetWrapper(inner, placeholder_tick_s=0.0)
    env.reset()                          # reset #1
    inner.next_done = True
    env.step(0)
    inner.gate.clear()
    env.reset()                          # async reset #2 pending
    inner.gate.set()
    obs, info = env.reset()              # explicit: join #2, then sync #3
    assert (obs == 3).all()
    assert "reset_pending" not in info
    env.close()


def test_one_envs_reset_does_not_stall_the_lockstep_batch():
    """The whole point of the feature: while env 0 runs a 1s reset, env 1
    keeps taking real steps -- 20 lockstep steps complete in well under the
    reset span, where today's synchronous auto-reset would block the batch
    for the full second on every one of env 0's episode ends."""
    vec = SubprocVecEnv([
        functools.partial(make_async_timed, reset_s=1.0, episode_len=2),
        functools.partial(make_async_timed, reset_s=0.0, episode_len=10 ** 9),
    ])
    try:
        vec.reset()
        started = time.monotonic()
        placeholders = 0
        for _ in range(20):
            _, _, _, infos = vec.step([0, 0])
            placeholders += bool(infos[0].get("reset_pending"))
        elapsed = time.monotonic() - started
        assert elapsed < 0.8, f"batch stalled: 20 steps took {elapsed:.2f}s"
        assert placeholders >= 1  # env 0 really was resetting during those steps
    finally:
        started = time.monotonic()
        vec.close()
        assert time.monotonic() - started < 5.0  # close never waits out a reset
