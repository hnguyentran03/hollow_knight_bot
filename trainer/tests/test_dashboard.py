import json
import os
import pathlib
import sys
import threading
import urllib.error
import urllib.request

import pytest

import hkrl.bosses
from hkrl.dashboard import make_server
from hkrl.fake_game import FakeGame, obs, state
from hkrl.generations import GenerationCallback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train  # noqa: E402  (path insert must precede this import)


def _won_episodes(n, steps=6):
    """Scripted wins, same shape as test_train's: known win_rate/damage."""
    ep = [state(obs(bhp=900)) for _ in range(steps - 1)]
    ep.append(state(obs(bhp=0), done=True, won=True))
    return [list(ep) for _ in range(n)]


@pytest.fixture()
def base_url(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "generations.jsonl").write_text(json.dumps(
        {"gen": 1, "timestep": 15_000, "wall_clock_s": 1000.0,
         "recoveries": 0, "episodes": 5, "mean_reward": 0.5,
         "win_rate": 0.0, "mean_episode_len": 500.0,
         "mean_boss_damage": 0.4}) + "\n")
    server = make_server(root=tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url, host=None):
    req = urllib.request.Request(url)
    if host is not None:
        req.add_unredirected_header("Host", host)
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


def test_serves_the_page_at_root(base_url):
    status, ctype, body = _get(base_url + "/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"<title>" in body


def test_api_runs_lists_run_summaries(base_url):
    status, ctype, body = _get(base_url + "/api/runs")
    assert status == 200
    assert ctype.startswith("application/json")
    runs = json.loads(body)
    assert [r["id"] for r in runs] == ["r1"]
    assert runs[0]["timestep"] == 15_000


def test_api_run_returns_the_full_series(base_url):
    _, _, body = _get(base_url + "/api/run/r1")
    run = json.loads(body)
    assert run["status"]["timestep"] == 15_000
    assert [g["gen"] for g in run["generations"]] == [1]
    # No exports dir yet: every generation reads unexported.
    assert [g["exported"] for g in run["generations"]] == [False]


def test_api_run_flags_exported_generations(base_url, tmp_path):
    exp = tmp_path / "exports" / "r1_gen0001"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"run_id": "r1", "gen": 1}))
    _, _, body = _get(base_url + "/api/run/r1")
    run = json.loads(body)
    assert [g["exported"] for g in run["generations"]] == [True]


def test_unknown_run_and_unknown_path_are_404(base_url):
    for path in ["/api/run/nope", "/favicon.ico"]:
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base_url + path)
        assert err.value.code == 404


def test_run_ids_cannot_escape_the_runs_dir(base_url):
    # %2e%2e = ".." once the handler unquotes; must not walk out of runs/.
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base_url + "/api/run/%2e%2e%2fr1")
    assert err.value.code == 404


def test_dashboard_serves_a_run_trained_on_two_instances(tmp_path):
    """The dashboard's whole contract with multi-instance training: the run
    directory a two-instance session writes (fleet-aggregated manifest, one
    VecMonitor file covering both envs) parses and serves like any other.
    Uses the real training stack against fake games, not hand-written
    fixtures, so a manifest/monitor format drift at N>1 fails here."""
    run_dir = tmp_path / "runs" / "multi"
    run_dir.mkdir(parents=True)
    with FakeGame(_won_episodes(40)) as a, FakeGame(_won_episodes(40)) as b:
        env, supervisor = train.build_env([a.port, b.port],
                                          relaunch=lambda s: None,
                                          run_dir=run_dir)
        try:
            model = train.build_model(env, run_dir, n_steps=8, batch_size=8)
            cb = GenerationCallback(run_dir, vecnorm=env, every_steps=16,
                                    supervisor=supervisor)
            model.learn(total_timesteps=32, callback=cb)
        finally:
            env.close()

    server = make_server(root=tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        runs = json.loads(_get(base + "/api/runs")[2])
        assert [r["id"] for r in runs] == ["multi"]
        run = json.loads(_get(base + "/api/run/multi")[2])
        assert run["status"]["timestep"] == 32
        assert [g["gen"] for g in run["generations"]] == [1, 2]
        assert all(g["win_rate"] == 1.0 for g in run["generations"])
    finally:
        server.shutdown()
        server.server_close()


import hkrl.dashboard as dash


def _post(url, body=None, ctype="application/json", host=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(),
                                 method="POST")
    req.add_header("Content-Type", ctype)
    if host is not None:
        req.add_unredirected_header("Host", host)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


def test_api_launcher_reports_idle_and_form_defaults(base_url, monkeypatch):
    monkeypatch.setattr(dash.launcher, "status", lambda root: None)
    _, _, body = _get(base_url + "/api/launcher")
    data = json.loads(body)
    assert data["active"] is None
    assert data["defaults"]["timesteps"] == 500_000
    assert data["defaults"]["instances"] == 1
    assert data["defaults"]["run_id"]


def test_api_launcher_reports_the_active_run(base_url, monkeypatch):
    rec = {"run_id": "r9", "pid": 4242, "started": 1000.0}
    monkeypatch.setattr(dash.launcher, "status", lambda root: rec)
    data = json.loads(_get(base_url + "/api/launcher")[2])
    assert data["active"] == rec


def test_api_launcher_lists_the_known_bosses(base_url):
    data = json.loads(_get(base_url + "/api/launcher")[2])
    bosses = data["bosses"]
    assert {b["id"] for b in bosses} == set(hkrl.bosses.BOSSES)
    assert [b["name"] for b in bosses] == sorted(b["name"] for b in bosses)
    by_id = {b["id"]: b["name"] for b in bosses}
    assert by_id["hornet1"] == "Hornet Protector"


def test_post_launch_delegates_and_returns_the_run_id(base_url, monkeypatch):
    seen = {}
    def fake_launch(root, params):
        seen.update(params)
        return "r9"
    monkeypatch.setattr(dash.launcher, "launch", fake_launch)
    status, data = _post(base_url + "/api/launch",
                         {"mode": "new", "run_id": "r9", "instances": 2})
    assert status == 200 and data == {"run_id": "r9"}
    assert seen["instances"] == 2


def test_post_launch_maps_errors_to_400_and_409(base_url, monkeypatch):
    def busy(root, params):
        raise RuntimeError("a launched run is already active")
    monkeypatch.setattr(dash.launcher, "launch", busy)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/launch", {})
    assert err.value.code == 409

    def bad(root, params):
        raise ValueError("instances must be between 1 and 3")
    monkeypatch.setattr(dash.launcher, "launch", bad)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/launch", {})
    assert err.value.code == 400
    assert "instances" in json.loads(err.value.read())["error"]


def test_post_stop_delegates_and_maps_idle_to_409(base_url, monkeypatch):
    monkeypatch.setattr(dash.launcher, "stop",
                        lambda root: {"run_id": "r9", "pid": 1, "started": 0})
    status, data = _post(base_url + "/api/stop")
    assert status == 200 and data == {"stopped": "r9"}

    def idle(root):
        raise RuntimeError("no launched run is active")
    monkeypatch.setattr(dash.launcher, "stop", idle)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/stop")
    assert err.value.code == 409


def test_posts_require_json_and_a_local_host_header(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/stop", ctype="text/plain")
    assert err.value.code == 415
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/stop", host="evil.example:9700")
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/nope")
    assert err.value.code == 404


def test_launcher_log_endpoint_tails_or_404s(base_url, monkeypatch):
    monkeypatch.setattr(dash.launcher, "tail",
                        lambda root, n=200: f"tail of {n}")
    status, ctype, body = _get(base_url + "/api/launcher/log?n=50")
    assert status == 200 and ctype.startswith("text/plain")
    assert body == b"tail of 50"

    monkeypatch.setattr(dash.launcher, "tail", lambda root, n=200: None)
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base_url + "/api/launcher/log")
    assert err.value.code == 404


def test_malformed_content_length_gets_a_400_not_a_dropped_socket(base_url):
    import socket
    host, port = base_url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall((f"POST /api/stop HTTP/1.1\r\n"
                   f"Host: 127.0.0.1:{port}\r\n"
                   "Content-Type: application/json\r\n"
                   "Content-Length: banana\r\n"
                   "Connection: close\r\n\r\n").encode())
        reply = s.recv(4096).decode(errors="replace")
    assert " 400 " in reply.splitlines()[0]


def test_api_gets_refuse_foreign_host(base_url):
    # DNS rebinding makes the attacker's page same-origin with us post
    # rebind, so a plain GET fetch would otherwise read run data straight
    # off the wire -- the Host header is what pins the request to us.
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base_url + "/api/runs", host="evil.example:9700")
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base_url + "/api/launcher/log", host="evil.example:9700")
    assert err.value.code == 403
    # A normal same-origin GET still works.
    status, _, _ = _get(base_url + "/api/runs")
    assert status == 200


def test_page_ships_the_launch_panel_and_summon_links(base_url):
    _, _, body = _get(base_url + "/")
    assert b'id="launch-form"' in body
    assert b"/api/launch" in body
    assert b'id="stop-btn"' in body
    assert b'id="summon-link"' in body
    assert b'id="prev-runs"' in body
    assert b'id="resume-dialog"' in body
    assert b'id="delete-dialog"' in body
    assert b'id="replay-dialog"' in body
    assert b'id="resume-btn"' not in body


def test_summon_serves_the_same_page_as_root(base_url):
    status_root, ctype_root, body_root = _get(base_url + "/")
    status, ctype, body = _get(base_url + "/summon")
    assert status == 200
    assert ctype.startswith("text/html")
    assert body == body_root


def test_post_replay_delegates_and_returns_run_and_gen(base_url, monkeypatch):
    seen = {}
    def fake_replay(root, run_id, gen, episodes):
        seen.update(run_id=run_id, gen=gen, episodes=episodes)
        return run_id
    monkeypatch.setattr(dash.launcher, "replay", fake_replay)
    status, data = _post(base_url + "/api/replay",
                         {"run_id": "r1", "gen": 2, "episodes": 5})
    assert status == 200 and data == {"replaying": "r1", "gen": 2}
    assert seen == {"run_id": "r1", "gen": 2, "episodes": 5}


def test_post_replay_defaults_episodes_and_maps_errors(base_url, monkeypatch):
    # episodes is optional; the handler defaults it to 3.
    seen = {}
    monkeypatch.setattr(dash.launcher, "replay",
                        lambda root, run_id, gen, episodes: seen.update(
                            episodes=episodes) or run_id)
    _post(base_url + "/api/replay", {"run_id": "r1", "gen": 1})
    assert seen["episodes"] == 3

    def busy(root, run_id, gen, episodes):
        raise RuntimeError("a launched run is already active")
    monkeypatch.setattr(dash.launcher, "replay", busy)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/replay", {"run_id": "r1", "gen": 1})
    assert err.value.code == 409

    def bad(root, run_id, gen, episodes):
        raise ValueError("gen must be a positive integer")
    monkeypatch.setattr(dash.launcher, "replay", bad)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/replay", {"run_id": "r1", "gen": 0})
    assert err.value.code == 400


def test_post_delete_delegates_and_maps_errors(base_url, monkeypatch):
    monkeypatch.setattr(dash.launcher, "delete",
                        lambda root, run_id: f"{run_id}-20260721_000000")
    status, data = _post(base_url + "/api/delete", {"run_id": "r1"})
    assert status == 200 and data == {"trashed": "r1-20260721_000000"}

    def busy(root, run_id):
        raise RuntimeError("'r1' is the active run; stop it first")
    monkeypatch.setattr(dash.launcher, "delete", busy)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/delete", {"run_id": "r1"})
    assert err.value.code == 409

    def gone(root, run_id):
        raise ValueError("no run named 'nope'")
    monkeypatch.setattr(dash.launcher, "delete", gone)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/delete", {"run_id": "nope"})
    assert err.value.code == 400


def _headers(url):
    with urllib.request.urlopen(url) as resp:
        return resp.headers


def test_polled_endpoints_are_not_cached(base_url):
    # The page polls these at fixed URLs every 2s; without an explicit
    # no-store, a real browser serves a stale cached body and the live
    # log/status freeze (Playwright disables its cache, hiding this).
    for path in ["/api/runs", "/api/run/r1", "/api/launcher"]:
        h = _headers(base_url + path)
        assert "no-store" in (h.get("Cache-Control") or ""), path


def test_post_export_delegates_and_returns_the_name(base_url, monkeypatch):
    calls = {}

    def fake_export(root, run_id, gen, name=None):
        calls.update(run_id=run_id, gen=gen, name=name)
        return "r1_gen0001"

    monkeypatch.setattr("hkrl.launcher.export", fake_export)
    status, data = _post(base_url + "/api/export", {"run_id": "r1", "gen": 1})
    assert status == 200
    assert data == {"exported": "r1_gen0001"}
    assert calls == {"run_id": "r1", "gen": 1, "name": None}


def test_post_export_maps_errors_to_400(base_url, monkeypatch):
    def bad(root, run_id, gen, name=None):
        raise ValueError("no run named 'r9'")

    monkeypatch.setattr("hkrl.launcher.export", bad)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/export", {"run_id": "r9", "gen": 1})
    assert err.value.code == 400


def test_page_wires_the_export_button(base_url):
    _, _, body = _get(base_url + "/")
    assert b"/api/export" in body


def test_api_launcher_reports_and_dismisses_the_last_exit(base_url,
                                                          tmp_path):
    d = tmp_path / "launcher"
    d.mkdir(exist_ok=True)
    (d / "r1.exit").write_text(json.dumps(
        {"run_id": "r1", "exit_code": 1, "ended_at": 5.0,
         "reason": ["startup failed: squatted"], "log_excerpt": "tail"}))
    _, _, body = _get(base_url + "/api/launcher")
    assert json.loads(body)["last_exit"]["reason"] == [
        "startup failed: squatted"]
    status, data = _post(base_url + "/api/launcher/dismiss-exit", {})
    assert status == 200 and data == {"dismissed": True}
    _, _, body = _get(base_url + "/api/launcher")
    assert json.loads(body)["last_exit"] is None


def _fake_active(root, run_id="r1"):
    """A pidfile naming this test process, so launcher.status() sees an
    active run without spawning anything."""
    d = root / "launcher"
    d.mkdir(exist_ok=True)
    (d / f"{run_id}.pid").write_text(json.dumps(
        {"run_id": run_id, "pid": os.getpid(), "started": 1000.0}))


def test_api_launcher_attaches_the_active_runs_config(base_url, tmp_path):
    (tmp_path / "runs" / "r1" / "config.jsonl").write_text(json.dumps(
        {"boss": "gruz_mother", "instances": 2, "headless": True,
         "timescale": 2.0, "timesteps": 500_000, "target_timestep": 1_000_000,
         "target_kl": 0.03, "port": 9020, "seed": None, "auto": True}) + "\n")
    _fake_active(tmp_path)
    _, _, body = _get(base_url + "/api/launcher")
    active = json.loads(body)["active"]
    assert active["run_id"] == "r1"
    assert active["config"] == {
        "boss": "gruz_mother", "instances": 2, "headless": True,
        "timescale": 2.0, "timesteps": 500_000,
        "target_timestep": 1_000_000, "target_kl": 0.03}


def test_api_launcher_config_is_null_before_train_py_writes_it(base_url,
                                                               tmp_path):
    # The launcher spawns train.py moments before prepare_session appends
    # config.jsonl; the first poll can land in that window.
    (tmp_path / "runs" / "r2").mkdir()
    _fake_active(tmp_path, "r2")
    _, _, body = _get(base_url + "/api/launcher")
    assert json.loads(body)["active"]["config"] is None


def test_api_launcher_config_survives_a_torn_tail(base_url, tmp_path):
    (tmp_path / "runs" / "r1" / "config.jsonl").write_text(
        json.dumps({"boss": "hornet1", "instances": 1}) + "\n"
        + '{"boss": "gruz_mo')  # torn mid-append
    _fake_active(tmp_path)
    _, _, body = _get(base_url + "/api/launcher")
    assert json.loads(body)["active"]["config"] == {
        "boss": "hornet1", "instances": 1}


def test_api_launcher_without_an_active_run_has_no_config(base_url):
    _, _, body = _get(base_url + "/api/launcher")
    assert json.loads(body)["active"] is None
