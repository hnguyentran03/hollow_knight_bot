import threading
import time

import numpy as np
import pytest

from hkrl.env import ACTIONS, DEFAULT_REWARD, HKEnv
from hkrl.fake_game import FakeGame, obs, state
from hkrl.protocol import ConnectionClosed
from hkrl.reset_metrics import read_reset_spans, reset_log_path


def test_action_space_and_obs_shape():
    episode = [state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, info = env.reset()
        assert env.action_space.n == len(ACTIONS) == 21
        assert o.shape == env.observation_space.shape
        assert o.dtype == np.float32
        env.close()


def test_reset_logs_a_span_per_reset_when_a_log_dir_is_configured(tmp_path):
    # Phase 0 instrumentation: each reset() records its wall-clock span to a
    # per-port sidecar, so a multi-instance run's sibling-freeze cost can be
    # measured after the fact.
    episodes = [[state(obs())], [state(obs())]]
    with FakeGame(episodes) as fg:
        env = HKEnv(port=fg.port, reset_log_dir=tmp_path)
        env.reset()
        env.reset()
        env.close()
    assert reset_log_path(tmp_path, fg.port).exists()
    spans = read_reset_spans(tmp_path)
    assert len(spans) == 2
    assert all(s >= 0.0 for s in spans)


def test_reset_does_not_log_without_a_log_dir(tmp_path):
    # Default (N=1, tests, non-measurement runs): no sidecar, no overhead.
    with FakeGame([[state(obs())]]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        env.close()
    assert read_reset_spans(tmp_path) == []


def test_reward_for_boss_damage_and_knight_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=880)), state(obs(bhp=880, khp=8))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, r1, *_ = env.step(0)
        assert r1 == (20 * DEFAULT_REWARD["boss_hp_scale"]
                      + DEFAULT_REWARD["time_penalty"])
        _, r2, *_ = env.step(0)
        assert r2 == (DEFAULT_REWARD["knight_hit"] + DEFAULT_REWARD["time_penalty"])
        env.close()


def test_win_and_loss_bonuses_and_termination():
    win_ep = [state(obs()), state(obs(bhp=0), done=True, won=True)]
    loss_ep = [state(obs()), state(obs(khp=0), done=True, won=False)]
    with FakeGame([win_ep, loss_ep]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, r, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and info["won"]
        assert r > DEFAULT_REWARD["win"] / 2  # dominated by the win bonus
        env.reset()
        _, r, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and not info["won"]
        assert r < 0
        env.close()


def test_unknown_boss_state_maps_to_fallback_slot():
    episode = [state(obs(boss_state="Some Brand New Move")), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, _ = env.reset()
        assert not np.isnan(o).any()
        env.close()


def test_truncation_at_max_steps():
    # Episode that never ends (no done=True), but we truncate at max_steps.
    # This guards against the bug where truncated timeout is incorrectly
    # reported as terminated, which would corrupt PPO's value estimates.
    episode = [state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        env.reset()
        # After one step, we should hit max_steps truncation.
        _, _, terminated, truncated, info = env.step(0)
        assert truncated is True, "truncated should be True when max_steps is reached"
        assert terminated is False, "terminated should be False for a truncation (not a real terminal state)"
        env.close()


def test_terminal_step_reports_boss_damage_fraction():
    episode = [state(obs(bhp=900)), state(obs(bhp=675), done=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, done, _, info = env.step(0)
        env.close()
    assert done is True
    assert info["boss_damage_frac"] == pytest.approx(0.25)


def test_win_reports_full_boss_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=0), done=True, won=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, _, _, info = env.step(0)
        env.close()
    assert info["boss_damage_frac"] == pytest.approx(1.0)


def test_mid_episode_steps_do_not_carry_boss_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=800)), state(obs(bhp=700), done=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, _, _, mid_info = env.step(0)
        _, _, _, _, end_info = env.step(0)
        env.close()
    assert "boss_damage_frac" not in mid_info
    assert end_info["boss_damage_frac"] == pytest.approx(200 / 900)


def test_truncation_reports_boss_damage_dealt_so_far():
    # No done=True frame: max_steps=1 forces truncation after one step, and
    # the damage dealt up to that point must still be reported.
    episode = [state(obs(bhp=900)), state(obs(bhp=450))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        env.reset()
        _, _, done, truncated, info = env.step(0)
        env.close()
    assert done is False and truncated is True
    assert info["boss_damage_frac"] == pytest.approx(0.5)


def test_reset_reconnects_through_the_mods_budget_expiry_drops():
    """The mod's 22.5s reset budget is deliberately smaller than a cold
    boot-to-fight, so each expiry drops the connection and the NEXT reset
    ratchets forward (title menu -> bench -> statue -> fight). Those drops
    are part of the protocol's normal rhythm and must be absorbed here by
    reconnecting and re-sending reset; if one escapes, the vec worker dies
    and the supervisor spends a whole rebuild -- or a relaunch-and-reboot --
    on a game that was healthy and mid-boot."""
    episode = [state(obs(khp=3)), state(obs())]
    with FakeGame([episode], fail_resets=2) as fg:
        env = HKEnv(port=fg.port)
        o, info = env.reset()
        # It is the post-ratchet reset's own first frame, not a stand-in.
        assert o[4] == pytest.approx(3 / 9)
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        env.close()


def test_reset_retries_are_bounded_so_a_drop_loop_still_surfaces():
    """A game that drops every reset forever (a genuinely broken boot) must
    still escalate to the supervisor rather than retrying silently all
    night."""
    with FakeGame([[state(obs())]], fail_resets=99) as fg:
        env = HKEnv(port=fg.port, reset_retries=2)
        with pytest.raises(ConnectionClosed):
            env.reset()
        env.close()


def test_keepalive_pings_keep_an_idle_connection_alive_and_invisible():
    """The N>1 starvation fix (hkrl/protocol.py Connection keepalive): while
    the trainer idles between lockstep messages -- another slot resetting, a
    PPO update -- the pinger must keep this connection's traffic flowing,
    and the pong replies must be invisible to step(). Interval shrunk to
    50ms so 300ms of idling stands in for a sibling's multi-second reset."""
    import time

    episode = [state(obs()), state(obs()), state(obs(bhp=0), done=True, won=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, keepalive=0.05)
        env.reset()
        time.sleep(0.3)  # several keepalive intervals of trainer silence
        assert fg.pings > 0  # the pinger really ran against the idle gap
        # The queued pongs sit between this step's action and its state;
        # recv() must filter them or this returns garbage/desyncs.
        _, _, terminated, *_ = env.step(0)
        assert not terminated
        _, _, terminated, *_ = env.step(0)
        assert terminated  # lockstep survived the pong interleaving intact
        env.close()


def test_abort_reset_unblocks_a_hung_reset_from_another_thread():
    """The async-reset shutdown path: a reset parked in its blocking recv
    (hang_resets: the fake accepts the reset request but never answers) must
    be releasable by another thread far inside the 30s socket timeout, and
    must re-raise instead of reconnecting."""
    errors = []
    with FakeGame([[state(obs())]], hang_resets=1) as fg:
        env = HKEnv(port=fg.port)

        def run():
            try:
                env.reset()
            except Exception as exc:  # noqa: BLE001 -- the raise IS the assertion
                errors.append(exc)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.2)          # let the reset park in its blocking recv
        started = time.monotonic()
        env.abort_reset()
        t.join(timeout=2.0)      # far under the 30s socket timeout
        assert not t.is_alive()
        assert time.monotonic() - started < 2.0
        assert errors and isinstance(errors[0], ConnectionClosed)
        env.close()
