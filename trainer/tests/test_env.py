import threading
import time

import numpy as np
import pytest

from hkrl.env import ACTIONS, DEFAULT_REWARD, HKEnv, StuckBoot, WrongSaveBoot
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


def test_win_bonus_scales_with_masks_remaining():
    # Winning with 7 of 9 masks left pays the flat win bonus plus
    # health_bonus per remaining mask (spec: docs/superpowers/specs/
    # 2026-07-29-value-reshaping-design.md).
    episode = [state(obs(bhp=100, khp=7)),
               state(obs(bhp=0, khp=7), done=True, won=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, r, terminated, _, info = env.step(0)
        env.close()
    assert terminated and info["won"]
    expected = (DEFAULT_REWARD["time_penalty"]
                + 100 * DEFAULT_REWARD["boss_hp_scale"]
                + DEFAULT_REWARD["win"]
                + 7 * DEFAULT_REWARD["health_bonus"])
    assert r == pytest.approx(expected)


def test_unknown_boss_state_maps_to_fallback_slot():
    episode = [state(obs(boss_state="Some Brand New Move")), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, _ = env.reset()
        assert not np.isnan(o).any()
        env.close()


def test_unseen_boss_state_warns_once_per_state(capfd):
    episode = [state(obs()), state(obs(boss_state="Gruz Slam")),
               state(obs(boss_state="Gruz Slam"))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        env.step(0)
        env.step(0)
        env.close()
    err = capfd.readouterr().err
    assert err.count("Gruz Slam") == 1
    assert "UNKNOWN" in err


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


def test_truncation_applies_the_death_penalty():
    # Running out the clock is a loss: the truncation step's reward carries
    # the death penalty, closing the stall-out loophole where a timeout was
    # cheaper (~-2.7 accumulated time penalty) than dying (-5).
    episode = [state(obs(bhp=900)), state(obs(bhp=900))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        env.reset()
        _, r, terminated, truncated, _ = env.step(0)
        env.close()
    assert truncated and not terminated
    assert r == pytest.approx(DEFAULT_REWARD["time_penalty"]
                              + DEFAULT_REWARD["death"])


def test_step_before_max_steps_carries_no_terminal_term():
    # One step short of the ceiling is still an ordinary step: time penalty
    # only, no death term leaking in early.
    episode = [state(obs()), state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=2)
        env.reset()
        _, r, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        assert r == pytest.approx(DEFAULT_REWARD["time_penalty"])
        env.close()


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


def test_abort_during_the_reconnect_window_still_aborts_promptly(monkeypatch):
    """abort_reset() racing the retry loop's close()/connect() reconnect: the
    flag can be set after the drop was caught but before/while the fresh
    connection comes up. The post-reconnect check must honor it instead of
    retrying against the new socket (where the abort's shutdown hit only the
    old, dead one)."""
    with FakeGame([[state(obs())], [state(obs())]], fail_resets=1) as fg:
        env = HKEnv(port=fg.port)
        original_connect = env.conn.connect

        def connect_then_abort():
            original_connect()
            env._reset_abort.set()  # abort lands exactly at the window's edge

        monkeypatch.setattr(env.conn, "connect", connect_then_abort)
        started = time.monotonic()
        with pytest.raises(ConnectionClosed):
            env.reset()
        assert time.monotonic() - started < 2.0
        env.close()


def test_env_rejects_an_unknown_boss_before_connecting():
    # No FakeGame: the registry lookup must fail before any socket work.
    with pytest.raises(ValueError, match="hornet1"):
        HKEnv(port=1, boss="grimm")


def test_obs_size_is_scalar_block_plus_boss_state_onehot():
    from hkrl.bosses import get_boss
    from hkrl.env import OBS_KEYS
    episode = [state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)   # default boss: hornet1
        n = len(OBS_KEYS) + len(get_boss("hornet1").fsm_states)
        assert env.observation_space.shape == (n,)
        env.close()


def test_reset_sends_the_boss_id():
    with FakeGame([[state(obs())]]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        assert fg.reset_bosses == ["hornet1"]
        env.close()


def test_old_mod_version_is_refused_at_connect():
    with FakeGame([[state(obs())]], version=1) as fg:
        with pytest.raises(RuntimeError, match="protocol"):
            HKEnv(port=fg.port)


def test_mod_error_reply_fails_the_reset_loudly():
    # A mod that doesn't know the requested boss answers with an error
    # instead of a state; that must raise, not retry or hang.
    with FakeGame([[state(obs())]], bosses=("gruz_mother",)) as fg:
        env = HKEnv(port=fg.port)
        with pytest.raises(RuntimeError, match="hornet1"):
            env.reset()
        env.close()


def test_fake_scene_follows_the_requested_boss():
    # The fake's synthetic scene is derived from the reset's boss id, so
    # scripted episodes stay boss-agnostic (and GG_-prefixed, which the
    # wrong-save classifier treats as Godhome-normal).
    ep = [state(obs())]
    with FakeGame([ep], bosses=("gruz_mother",)) as fg:
        env = HKEnv(port=fg.port, boss="gruz_mother")
        _, info = env.reset()
    assert info["scene"] == "GG_gruz_mother"


def test_two_consecutive_wrong_scene_aborts_raise_wrong_save_boot():
    ep = [state(obs())]
    with FakeGame([ep], abort_scenes=["Tutorial_01", "Tutorial_01"]) as fg:
        env = HKEnv(port=fg.port)
        with pytest.raises(WrongSaveBoot, match="Tutorial_01"):
            env.reset()


def test_menu_and_godhome_aborts_never_trip_and_reset_the_streak():
    # Wrong, then Godhome (streak back to 0), then wrong again, then menu
    # (streak back to 0 again): never reaches 2 consecutive, and the 5th
    # attempt's clean reset succeeds. Exercises both whitelist families
    # (GG_* and Menu_*), not just one.
    ep = [state(obs())]
    scenes = ["Tutorial_01", "GG_Workshop", "Tutorial_01", "Menu_Title"]
    with FakeGame([ep], abort_scenes=scenes) as fg:
        env = HKEnv(port=fg.port)
        o, info = env.reset()
    assert info["scene"].startswith("GG_")


def test_wrong_save_boot_is_recoverable_for_the_supervisor():
    # The supervisor's RECOVERABLE handling keys off ConnectionClosed-shaped
    # failures; WrongSaveBoot must stay inside that family.
    assert issubclass(WrongSaveBoot, ConnectionClosed)


def test_five_consecutive_aborts_anywhere_raise_stuck_boot():
    # A corrupt (not missing) save renders the slot unselectable: the boot
    # macro stalls at save select in Menu_Title, which the wrong-save
    # whitelist ignores. The total-abort streak catches the stall.
    ep = [state(obs())]
    with FakeGame([ep], abort_scenes=["Menu_Title"] * 5) as fg:
        env = HKEnv(port=fg.port)
        with pytest.raises(StuckBoot, match="Menu_Title"):
            env.reset()


def test_four_aborts_then_success_never_trip_stuck_boot():
    ep = [state(obs())]
    with FakeGame([ep], abort_scenes=["Menu_Title"] * 4) as fg:
        env = HKEnv(port=fg.port)
        o, info = env.reset()
    assert info["scene"].startswith("GG_")


def test_stuck_boot_is_recoverable_for_the_supervisor():
    assert issubclass(StuckBoot, ConnectionClosed)
