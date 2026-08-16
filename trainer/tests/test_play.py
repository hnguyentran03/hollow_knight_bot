import json
import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import play  # noqa: E402  (path insert must precede this import)

from hkrl.protocol import ConnectionClosed  # noqa: E402


# ---- stubs ----

class StubModel:
    def __init__(self):
        self.starts = []

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        self.starts.append(bool(episode_start[0]))
        return np.array([0]), state


class StubVecnorm:
    def normalize_obs(self, o):
        return o


class ScriptedConn:
    """recv() yields scripted messages; an Exception instance is raised."""

    def __init__(self, events):
        self.events = list(events)
        self.accept_events = False

    def recv(self):
        e = self.events.pop(0)
        # BaseException, not Exception: the loop-terminating
        # KeyboardInterrupt the tests script is NOT an Exception subclass.
        if isinstance(e, BaseException):
            raise e
        return e


class StubEnv:
    def __init__(self, steps=2, conn=None):
        self.reset_calls = 0
        self.boss_ids = []
        self.conn = conn or types.SimpleNamespace(accept_events=False)
        self._steps = steps
        self._n = 0

    def set_boss(self, boss_id):
        if boss_id == "unknowable":
            raise ValueError(f"unknown boss {boss_id!r}")
        self.boss_ids.append(boss_id)

    def reset(self):
        self.reset_calls += 1
        self._n = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._n += 1
        done = self._n >= self._steps
        info = {"won": True, "boss_damage_frac": 1.0} if done else {}
        return np.zeros(3, dtype=np.float32), 1.0, done, False, info


def _export(tmp_path, name="bot1", boss="gruz_mother"):
    d = tmp_path / "exports" / name
    d.mkdir(parents=True)
    (d / "model.zip").write_bytes(b"m")
    (d / "vecnorm.pkl").write_bytes(b"v")
    manifest = {"name": name, "gen": 4, "run_id": "r1",
                "boss_display": "Gruz Mother",
                "stats": {"win_rate": 0.8}}
    if boss is not None:
        manifest["boss"] = boss
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def _loaders():
    calls = {"model": 0, "vecnorm": 0}

    def load_model(path):
        calls["model"] += 1
        return StubModel()

    def load_vecnorm(path):
        calls["vecnorm"] += 1
        return StubVecnorm()

    return calls, load_model, load_vecnorm


# ---- load_export ----

def test_load_export_reads_manifest_and_caches(tmp_path):
    _export(tmp_path)
    calls, lm, lv = _loaders()
    cache = {}
    entry = play.load_export(tmp_path, "bot1", cache,
                             load_model=lm, load_vecnorm=lv)
    assert entry["manifest"]["boss"] == "gruz_mother"
    again = play.load_export(tmp_path, "bot1", cache,
                             load_model=lm, load_vecnorm=lv)
    assert again is entry
    assert calls == {"model": 1, "vecnorm": 1}   # loaded exactly once


def test_load_export_missing_dir_names_the_root(tmp_path):
    with pytest.raises(play.ExportError, match="exports"):
        play.load_export(tmp_path, "ghost", {},
                         load_model=None, load_vecnorm=None)


def test_load_export_unreadable_manifest_is_an_exporterror(tmp_path):
    d = _export(tmp_path)
    (d / "manifest.json").write_text("{not json")
    with pytest.raises(play.ExportError, match="manifest"):
        play.load_export(tmp_path, "bot1", {},
                         load_model=None, load_vecnorm=None)


# ---- play_episode ----

def test_play_episode_runs_one_episode_no_trailing_reset(capsys):
    env, model = StubEnv(steps=3), StubModel()
    summary = play.play_episode(env, model, StubVecnorm(), name="bot1")
    assert env.reset_calls == 1          # exactly one reset, sent up front
    assert summary["result"] == "WIN" and summary["steps"] == 3
    assert model.starts == [True, False, False]  # LSTM state init pattern
    out = capsys.readouterr().out
    assert "bot1" in out and "result=WIN" in out


# ---- handle_event ----

def _play_event(bot="bot1"):
    return {"type": "event", "name": "play", "bot": bot}


def test_handle_event_plays_the_named_bot(tmp_path, capsys):
    _export(tmp_path)
    calls, lm, lv = _loaders()
    env = StubEnv()
    summary = play.handle_event(_play_event(), root=tmp_path, env=env,
                                cache={}, load_model=lm, load_vecnorm=lv)
    assert summary["result"] == "WIN"
    assert env.boss_ids == ["gruz_mother"]   # boss came from the manifest
    assert env.reset_calls == 1
    assert "bot1" in capsys.readouterr().out  # banner + summary line


def test_handle_event_ignores_non_play_messages(tmp_path):
    env = StubEnv()
    assert play.handle_event({"type": "state"}, root=tmp_path, env=env,
                             cache={}) is None
    assert play.handle_event({"type": "event", "name": "other"},
                             root=tmp_path, env=env, cache={}) is None
    assert env.reset_calls == 0


def test_handle_event_survives_a_missing_export(tmp_path, capsys):
    env = StubEnv()
    assert play.handle_event(_play_event("ghost"), root=tmp_path, env=env,
                             cache={}) is None
    assert env.reset_calls == 0
    assert "ghost" in capsys.readouterr().err


def test_handle_event_survives_a_manifest_without_boss(tmp_path, capsys):
    _export(tmp_path, boss=None)
    calls, lm, lv = _loaders()
    env = StubEnv()
    assert play.handle_event(_play_event(), root=tmp_path, env=env,
                             cache={}, load_model=lm, load_vecnorm=lv) is None
    assert env.reset_calls == 0
    assert "boss" in capsys.readouterr().err


def test_handle_event_survives_an_unknown_boss(tmp_path, capsys):
    _export(tmp_path, boss="unknowable")
    calls, lm, lv = _loaders()
    env = StubEnv()
    assert play.handle_event(_play_event(), root=tmp_path, env=env,
                             cache={}, load_model=lm, load_vecnorm=lv) is None
    assert env.reset_calls == 0
    assert "unknown boss" in capsys.readouterr().err


def test_handle_event_without_a_bot_name(tmp_path, capsys):
    env = StubEnv()
    assert play.handle_event({"type": "event", "name": "play"},
                             root=tmp_path, env=env, cache={}) is None
    assert "selected" in capsys.readouterr().err


# ---- idle_loop ----

def test_idle_loop_dispatches_and_opts_into_events(tmp_path):
    _export(tmp_path)
    calls, lm, lv = _loaders()
    conn = ScriptedConn([TimeoutError(), _play_event(),
                         KeyboardInterrupt()])
    env = StubEnv(conn=conn)
    with pytest.raises(KeyboardInterrupt):
        play.idle_loop(env, tmp_path, {}, load_model=lm, load_vecnorm=lv)
    assert conn.accept_events is True
    assert env.reset_calls == 1              # timeout looped, event played


def test_idle_loop_reconnects_on_a_closed_connection(tmp_path):
    conn = ScriptedConn([ConnectionClosed("gone"), KeyboardInterrupt()])
    env = StubEnv(conn=conn)
    reconnects = []
    with pytest.raises(KeyboardInterrupt):
        play.idle_loop(env, tmp_path, {},
                       reconnect=lambda e, out: reconnects.append(e))
    assert reconnects == [env]


# ---- parser ----

def test_parser_has_no_bot_argument_and_sane_defaults():
    args = play.build_parser().parse_args([])
    assert args.port == 9020
    assert args.host == "127.0.0.1"
    assert args.stochastic is False
    assert args.root == pathlib.Path("~/hkrl").expanduser()
    with pytest.raises(SystemExit):
        play.build_parser().parse_args(["somebot"])   # menu-only selection
