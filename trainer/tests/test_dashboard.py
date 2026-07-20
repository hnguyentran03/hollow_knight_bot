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
