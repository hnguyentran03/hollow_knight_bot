import json
import threading
import urllib.error
import urllib.request

import pytest

from hkrl.dashboard import make_server


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
