import json
import pathlib
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import random_agent  # noqa: E402  (path insert must precede this import)

from hkrl.env import HKEnv
from hkrl.fake_game import FakeGame


def obs(kx=20.0, khp=9, bhp=900, boss_state="Idle", **kw):
    base = {"kx": kx, "ky": 6.0, "kvx": 0.0, "kvy": 0.0, "khp": khp, "soul": 0,
            "on_ground": True, "dashing": False, "invuln": False, "facing_right": True,
            "bx": 30.0, "by": 6.0, "bvx": 0.0, "bvy": 0.0, "bhp": bhp,
            "boss_state": boss_state, "needle_active": False, "nx": 0.0, "ny": 0.0}
    base.update(kw)
    return base


def state(o, done=False, won=False, attempt=1, scene="GG_Hornet_1"):
    return {"type": "state", "obs": o, "done": done,
            "info": {"won": won, "scene": scene, "attempt": attempt}}


def test_run_completes_scripted_episodes_and_reports_summaries(capsys):
    win_ep = [state(obs(bhp=900), attempt=1), state(obs(bhp=0), done=True, won=True, attempt=1)]
    loss_ep = [state(obs(khp=9), attempt=2), state(obs(khp=0), done=True, won=False, attempt=2)]
    with FakeGame([win_ep, loss_ep]) as fg:
        env = HKEnv(port=fg.port)
        summaries = random_agent.run(env, episodes=2)
        env.close()

    assert len(summaries) == 2
    assert summaries[0]["attempt"] == 1 and summaries[0]["won"] is True
    assert summaries[1]["attempt"] == 2 and summaries[1]["won"] is False

    out = capsys.readouterr().out
    assert "episode   1 (attempt 1)" in out
    assert "episode   2 (attempt 2)" in out
    assert "result=WIN" in out
    assert "result=loss" in out
    # required minimum fields per the task brief: attempt, steps, reward, won
    assert "steps=" in out and "reward=" in out


def test_truncation_reported_distinctly_from_termination(capsys):
    # No state ever sets done=True: the env's own max_steps=1 forces a
    # truncation after the first step. This must show up as TIMEOUT, not
    # get confused with a real win/loss.
    episode = [state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        summaries = random_agent.run(env, episodes=1)
        env.close()

    assert summaries[0]["truncated"] is True
    assert summaries[0]["terminated"] is False
    out = capsys.readouterr().out
    assert "result=TIMEOUT" in out


def test_warns_when_attempt_does_not_advance(capsys):
    # Two episodes that both report attempt=1: a genuine reset should bump
    # the mod's attempt counter, so a repeat is a red flag that the reset
    # didn't actually happen even though the episode "completed". Each
    # episode needs its own reset reply (non-terminal) plus a terminal step
    # reply -- reset() never itself carries done=True, per the protocol.
    ep1 = [state(obs(bhp=900), attempt=1), state(obs(bhp=0), done=True, won=True, attempt=1)]
    ep2 = [state(obs(bhp=900), attempt=1), state(obs(bhp=0), done=True, won=True, attempt=1)]
    with FakeGame([ep1, ep2]) as fg:
        env = HKEnv(port=fg.port)
        random_agent.run(env, episodes=2)
        env.close()

    out = capsys.readouterr().out
    assert "WARNING: attempt did not advance" in out


def _serve_then_disconnect():
    """A minimal hand-rolled server (like test_protocol.py's _serve) that
    replies to reset() once, then cleanly closes the connection the moment
    it reads the first action -- simulating the mod crashing/disconnecting
    mid-episode.

    Deliberately NOT built on FakeGame's own episode-exhaustion path (an
    empty episode list makes its worker thread raise an unhandled
    IndexError, caught only by its `finally: conn.close()`): under pytest,
    letting that exception escape the thread and reach pytest's own
    threading.excepthook makes the *client's* blocking read wait out the
    full 30s protocol socket timeout before ever observing the close, even
    though the close() call itself completes in microseconds (confirmed by
    isolated repro -- a clean, exception-free close is detected by the
    client instantly). That is a pytest/threading interaction quirk, not
    something intrinsic to a real mod disconnect, so it is avoided here
    rather than exercised, to keep this test fast and reliable.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        f = conn.makefile("rwb")

        def send(msg):
            f.write(json.dumps(msg).encode() + b"\n")
            f.flush()

        send({"type": "hello", "version": 1})
        f.readline()  # reset
        send(state(obs()))
        f.readline()  # action -- disconnect instead of replying
        conn.close()
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_wedge_raised_and_named_when_mod_disconnects_mid_episode():
    port = _serve_then_disconnect()
    env = HKEnv(port=port)
    with pytest.raises(random_agent.Wedge, match="episode 1"):
        random_agent.run_episode(env, 1)
    env.close()


def _slow_action_server(delay):
    """A minimal hand-rolled server (like test_protocol.py's _serve) that
    replies to reset() immediately but sleeps `delay` seconds before
    replying to the first action -- used to exercise the heartbeat without
    waiting anywhere near the real 30s socket timeout.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        f = conn.makefile("rwb")

        def send(msg):
            f.write(json.dumps(msg).encode() + b"\n")
            f.flush()

        send({"type": "hello", "version": 1})
        f.readline()  # reset
        send(state(obs(bhp=900)))
        f.readline()  # action
        time.sleep(delay)
        send(state(obs(bhp=0), done=True, won=True))
        conn.close()
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_heartbeat_prints_when_a_call_is_slow(capsys):
    port = _slow_action_server(delay=0.3)
    env = HKEnv(port=port)
    random_agent.run_episode(env, 1, step_warn_after=0.05, step_warn_every=0.05)
    env.close()

    err = capsys.readouterr().err
    assert "still waiting on step()" in err


def test_run_preserves_completed_summaries_when_a_later_episode_wedges(capsys):
    # A run() of 2 episodes where the first completes normally and the
    # second's server disconnects mid-episode. The completed episode must
    # not be silently dropped from the tally just because a later one
    # wedged -- Wedge.summaries carries whatever run() had already
    # accumulated (see Wedge's docstring in random_agent.py).
    win_ep = [state(obs(bhp=900), attempt=1), state(obs(bhp=0), done=True, won=True, attempt=1)]
    port = _serve_one_episode_then_disconnect(win_ep)
    env = HKEnv(port=port)
    with pytest.raises(random_agent.Wedge) as excinfo:
        random_agent.run(env, episodes=2)
    env.close()

    preserved = excinfo.value.summaries
    assert len(preserved) == 1
    assert preserved[0]["attempt"] == 1 and preserved[0]["won"] is True


def _serve_one_episode_then_disconnect(episode_messages):
    """Replays `episode_messages` in full for one reset/episode, then
    disconnects cleanly as soon as the client asks to reset again."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        f = conn.makefile("rwb")

        def send(msg):
            f.write(json.dumps(msg).encode() + b"\n")
            f.flush()

        send({"type": "hello", "version": 1})
        remaining = list(episode_messages)
        f.readline()  # reset
        send(remaining.pop(0))
        while remaining:
            f.readline()  # action
            send(remaining.pop(0))
        f.readline()  # second episode's reset -- disconnect instead
        conn.close()
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_friendly_error_when_mod_not_reachable(capsys):
    # F7: constructing HKEnv against a port nobody is listening on should
    # print a clear, human-readable message naming the likely cause (game
    # not running / mod not installed / wrong port) and exit(1), instead of
    # letting a raw ConnectionRefusedError traceback reach the user. Bind an
    # ephemeral port then immediately close it so nothing is listening --
    # this reproduces a real connection refusal without needing the game.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    old_argv = sys.argv
    sys.argv = ["random_agent.py", "--port", str(port), "--episodes", "1"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            random_agent.main()
    finally:
        sys.argv = old_argv

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "could not connect" in err.lower()
    assert "mod" in err.lower()


def test_no_heartbeat_for_normal_speed_episode(capsys):
    # Sanity check for the other half of "distinguishable from slow
    # progress": a normal-speed episode against FakeGame (no artificial
    # delay) must not print any heartbeat noise.
    episode = [state(obs(bhp=900)), state(obs(bhp=0), done=True, won=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        random_agent.run_episode(env, 1)
        env.close()

    err = capsys.readouterr().err
    assert "still waiting" not in err
