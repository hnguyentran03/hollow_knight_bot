import time

import pytest

from hkrl.async_reset import AsyncResetWrapper
from hkrl.fake_slow_env import SlowResetEnv


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
