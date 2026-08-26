import json
import pathlib
import sys

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import replay  # noqa: E402  (path insert must precede this import)

from hkrl.fake_game import FakeGame, obs, state
from hkrl.vec import make_env


def _scripted(win):
    """A 4-step episode with a known outcome: reset frame, three mid frames,
    then a terminal frame -- boss at 0 HP on a win, knight dead on a loss
    with half the boss's HP removed."""
    frames = [state(obs(bhp=900))] + [state(obs(bhp=900)) for _ in range(3)]
    if win:
        frames.append(state(obs(bhp=0), done=True, won=True))
    else:
        frames.append(state(obs(bhp=450, khp=0), done=True))
    return frames


def _make_checkpoint(tmp_path):
    """A real, loadable checkpoint: an untrained RecurrentPPO with this
    pipeline's spaces, plus the VecNormalize statistics it was constructed
    with. No VecFrameStack -- the recurrent policy replaced frame stacking,
    so replay's pipeline (which this must match) no longer stacks either."""
    with FakeGame([_scripted(win=False)]) as fg:
        venv = DummyVecEnv([make_env(fg.port)])
        env = VecNormalize(venv, gamma=0.995)
        model = RecurrentPPO("MlpLstmPolicy", env, n_steps=8, batch_size=8,
                             device="cpu")
        weights = tmp_path / "gen_0001.zip"
        vecnorm = tmp_path / "gen_0001_vecnorm.pkl"
        model.save(weights)
        env.save(str(vecnorm))
        env.close()
    return weights, vecnorm


def test_replay_reports_per_episode_stats(tmp_path, capsys):
    weights, vecnorm = _make_checkpoint(tmp_path)
    # One extra scripted episode: DummyVecEnv autoresets on the final terminal step, eagerly consuming one more reset() before replay()'s loop exits.
    with FakeGame([_scripted(win=True), _scripted(win=False), _scripted(win=False)]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path)
        try:
            summaries = replay.replay(model, env, episodes=2)
        finally:
            env.close()

    assert [s["result"] for s in summaries] == ["WIN", "loss"]
    assert summaries[0]["won"] is True
    assert summaries[0]["steps"] == 4
    assert summaries[0]["boss_damage_frac"] == pytest.approx(1.0)
    assert summaries[1]["boss_damage_frac"] == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "result=WIN" in out and "boss_dmg=" in out
    # Rewards are printed unnormalized: a win's +10 terminal bonus dominates.
    assert summaries[0]["reward"] > 5.0


def test_banner_names_generation_run_and_episodes():
    text = replay.banner(3, pathlib.Path("/x/runs/r1"), 5)
    assert "generation 3" in text
    assert "/x/runs/r1" in text
    assert "5 episodes" in text


def test_auto_and_root_are_accepted_flags():
    # --auto makes replay self-contained (launches its own game); --root is
    # where it backs up the master save. Both parse WITHOUT touching a game.
    args = replay.build_parser().parse_args(
        ["--run-dir", "/x", "--auto", "--gen", "2", "--episodes", "4",
         "--root", "/tmp/hk"])
    assert args.auto is True
    assert args.gen == 2 and args.episodes == 4
    assert str(args.root) == "/tmp/hk"


def test_auto_defaults_off_and_root_has_a_default():
    args = replay.build_parser().parse_args(["--run-dir", "/x"])
    assert args.auto is False        # unchanged: connect to a running game
    assert args.root is not None     # ~/hkrl by default, for backup_saves


def test_headless_and_timescale_are_accepted_flags():
    # --auto --headless --timescale K is the speed-fidelity gate command;
    # both must parse without touching a game.
    args = replay.build_parser().parse_args(
        ["--run-dir", "x", "--auto", "--headless", "--timescale", "2"])
    assert args.headless is True
    assert args.timescale == 2.0
    args = replay.build_parser().parse_args(["--run-dir", "x"])
    assert args.headless is False
    assert args.timescale == 1.0


def test_replay_stops_at_the_episode_boundary_when_flagged(tmp_path):
    # A set stop flag ends the loop at the next episode boundary (mirrors
    # train.py's StopOnFlag): the dashboard's single Stop -> SIGINT sets it,
    # and the in-progress episode still finishes rather than being severed.
    import threading
    weights, vecnorm = _make_checkpoint(tmp_path)
    stop = threading.Event()
    stop.set()
    with FakeGame([_scripted(win=True), _scripted(win=False)]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path)
        try:
            summaries = replay.replay(model, env, episodes=5, stop=stop)
        finally:
            env.close()
    assert summaries == []  # already stopped: not even the first episode ran


def test_deterministic_replay_reproduces_itself(tmp_path):
    # The frozen-statistics + deterministic-argmax pipeline must be
    # repeatable: identical scripted games produce identical summaries.
    weights, vecnorm = _make_checkpoint(tmp_path)
    results = []
    for _ in range(2):
        # One extra scripted episode: DummyVecEnv autoresets on the final terminal step, eagerly consuming one more reset() before replay()'s loop exits.
        with FakeGame([_scripted(win=False), _scripted(win=False)]) as fg:
            model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                            run_dir=tmp_path)
            try:
                results.append(replay.replay(model, env, episodes=1))
            finally:
                env.close()
    assert results[0] == results[1]


def test_run_boss_reads_config_and_defaults_to_hornet1(tmp_path):
    assert replay.run_boss(tmp_path) == "hornet1"          # no config at all
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother"}) + "\n")
    assert replay.run_boss(tmp_path) == "gruz_mother"


from hkrl.recording import RecordingWriter, read_recording


def _predict_actions(model, env, episodes):
    """The stock predict() loop, actions collected: the recorder must make
    identical deterministic choices or its state threading is wrong."""
    import numpy as np
    actions, obs = [], env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    for _ in range(episodes):
        done = False
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=True)
            actions.append(int(action[0]))
            obs, _, dones, _ = env.step(action)
            episode_starts = dones
            done = bool(dones[0])
    return actions


def _record(tmp_path, weights, vecnorm, scripts, episodes):
    with FakeGame([list(ep) for ep in scripts]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path, capture=True)
        writer = RecordingWriter(tmp_path / "rec.jsonl.gz")
        try:
            summaries = replay.record_replay(model, env, episodes=episodes,
                                             writer=writer)
        finally:
            writer.close()
            env.close()
    return summaries, read_recording(tmp_path / "rec.jsonl.gz")


def test_record_replay_actions_match_model_predict(tmp_path):
    weights, vecnorm = _make_checkpoint(tmp_path)
    scripts = [_scripted(win=True), _scripted(win=False), _scripted(win=False)]
    with FakeGame([list(ep) for ep in scripts]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path)
        try:
            expected = _predict_actions(model, env, episodes=2)
        finally:
            env.close()
    _, rows = _record(tmp_path, weights, vecnorm, scripts, episodes=2)
    steps = [r for r in rows if r["type"] == "step"]
    assert [r["a"] for r in steps] == expected


def test_record_replay_row_structure(tmp_path):
    import numpy as np
    weights, vecnorm = _make_checkpoint(tmp_path)
    scripts = [_scripted(win=True), _scripted(win=False), _scripted(win=False)]
    summaries, rows = _record(tmp_path, weights, vecnorm, scripts, episodes=2)
    steps = [r for r in rows if r["type"] == "step"]
    episodes = [r for r in rows if r["type"] == "episode"]
    # Interleaving: no header here (task 5 wires it); step counts match the
    # summaries; one episode line per episode, after its last step.
    assert len(episodes) == 2
    assert [e["steps"] for e in episodes] == [s["steps"] for s in summaries]
    assert len(steps) == sum(s["steps"] for s in summaries)
    assert rows[-1]["type"] == "episode"
    for r in steps:
        assert len(r["pi"]) == 21
        assert abs(sum(r["pi"]) - 1.0) < 1e-3          # 5-sig-digit rounding
        assert r["a"] == int(np.argmax(r["pi"]))       # deterministic mode
        assert set(r["obs"]) >= {"kx", "khp", "bhp", "boss_state"}
        # Loose tolerance on purpose: r rides DummyVecEnv's float32 reward
        # buffer while the term sum is float64, so a ~46.0 terminal reward
        # legitimately differs by ~3e-6. NOT a state-threading bug.
        assert sum(r["r_terms"].values()) == pytest.approx(r["r"], abs=1e-4)
        assert isinstance(r["v"], float) and isinstance(r["ent"], float)
    # Outcome flags live on each episode's last row.
    assert steps[-1]["done"] or steps[-1]["trunc"]
    assert episodes[0]["result"] == "WIN" and episodes[1]["result"] == "loss"
    assert episodes[0]["boss_damage_frac"] == pytest.approx(1.0)


def test_record_replay_episode_boundary_obs_pairing(tmp_path):
    # The autoreset trap (spec 4.5): episode 2's first row must hold the
    # fresh fight's frame from reset_infos, not episode 1's death frame.
    weights, vecnorm = _make_checkpoint(tmp_path)
    scripts = [_scripted(win=False), _scripted(win=True), _scripted(win=False)]
    _, rows = _record(tmp_path, weights, vecnorm, scripts, episodes=2)
    steps = [r for r in rows if r["type"] == "step"]
    ep1_last = [r for r in steps if r["ep"] == 1][-1]
    ep2_first = [r for r in steps if r["ep"] == 2][0]
    assert ep1_last["done"] and ep1_last["won"] is False
    assert ep2_first["i"] == 0
    assert ep2_first["obs"]["khp"] == 9      # fresh fight, not the khp=0 death frame
    assert ep2_first["obs"]["bhp"] == 900


from hkrl.recording import SCHEMA_VERSION


def test_parser_accepts_record_flags():
    args = replay.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--record", "--record-dir", "/tmp/out"])
    assert args.record is True
    assert str(args.record_dir) == "/tmp/out"
    defaults = replay.build_parser().parse_args(["--run-dir", "/tmp/x"])
    assert defaults.record is False and defaults.record_dir is None


def test_run_connected_records_a_complete_file(tmp_path):
    weights, vecnorm = _make_checkpoint(tmp_path)
    record = tmp_path / "replays" / "test_gen0001.jsonl.gz"
    scripts = [_scripted(win=True), _scripted(win=False)]
    with FakeGame([list(ep) for ep in scripts]) as fg:
        summaries = replay.run_connected(
            weights, vecnorm, run_dir=tmp_path, host="127.0.0.1",
            port=fg.port, episodes=1, deterministic=True,
            record=record, gen=1)
    assert [s["result"] for s in summaries] == ["WIN"]
    rows = read_recording(record)
    assert rows[0]["type"] == "header"
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["gen"] == 1 and rows[0]["boss"] == "hornet1"
    assert rows[0]["deterministic"] is True and rows[0]["auto"] is False
    assert [r["type"] for r in rows[1:]] == ["step"] * summaries[0]["steps"] + ["episode"]


def test_run_connected_without_record_behaves_as_before(tmp_path):
    weights, vecnorm = _make_checkpoint(tmp_path)
    scripts = [_scripted(win=True), _scripted(win=False)]
    with FakeGame([list(ep) for ep in scripts]) as fg:
        summaries = replay.run_connected(
            weights, vecnorm, run_dir=tmp_path, host="127.0.0.1",
            port=fg.port, episodes=1, deterministic=True)
    assert [s["result"] for s in summaries] == ["WIN"]
    assert not (tmp_path / "replays").exists()


def test_record_replay_truncation_rows_and_summary(tmp_path):
    # The capture branch deferred this: a fight that hits max_steps must
    # record trunc=True (done stays False), a TIMEOUT episode row with the
    # truncation death penalty in its final terms, and still pair the next
    # episode's first obs correctly across the autoreset.
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    weights, vecnorm = _make_checkpoint(tmp_path)
    # Six mid frames after the reset frame; max_steps=3 truncates first.
    long_ep = [state(obs(bhp=900)) for _ in range(7)]
    with FakeGame([long_ep, _scripted(win=True), _scripted(win=False)]) as fg:
        venv = DummyVecEnv([make_env(fg.port, capture=True, max_steps=3)])
        env = VecNormalize.load(str(vecnorm), venv)
        env.training = False
        env.norm_reward = False
        model = RecurrentPPO.load(str(weights), device="cpu")
        writer = RecordingWriter(tmp_path / "trunc.jsonl.gz")
        try:
            summaries = replay.record_replay(model, env, episodes=2,
                                             writer=writer)
        finally:
            writer.close()
            env.close()
    assert summaries[0]["result"] == "TIMEOUT"
    rows = read_recording(tmp_path / "trunc.jsonl.gz")
    steps = [r for r in rows if r["type"] == "step"]
    ep1_last = [r for r in steps if r["ep"] == 1][-1]
    assert ep1_last["trunc"] is True and ep1_last["done"] is False
    assert ep1_last["r_terms"].get("death") == -5.0
    episodes = [r for r in rows if r["type"] == "episode"]
    assert episodes[0]["result"] == "TIMEOUT"
    ep2_first = [r for r in steps if r["ep"] == 2][0]
    assert ep2_first["i"] == 0 and ep2_first["obs"]["bhp"] == 900
