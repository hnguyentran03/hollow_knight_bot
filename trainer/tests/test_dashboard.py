import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

import pytest

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


def _get(url):
    with urllib.request.urlopen(url) as resp:
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
