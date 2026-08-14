import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train  # noqa: E402  (path insert must precede this import)

import pytest  # noqa: E402

from hkrl.fake_game import FakeGame, obs, state
from hkrl.generations import GenerationCallback, latest_checkpoint
from hkrl.reset_metrics import read_reset_spans
from hkrl.masking import MaskedRecurrentPPO, MaskedRecurrentRolloutBuffer
from hkrl.vec import RealEpisodeVecMonitor, RealEpisodeVecNormalize


def _won_episode(steps=6):
    """Every scripted episode ends in a win with the boss at 0 HP, so the
    manifest's win_rate and mean_boss_damage have known expected values."""
    frames = [state(obs(bhp=900))]
    frames += [state(obs(bhp=900)) for _ in range(steps - 1)]
    frames.append(state(obs(bhp=0), done=True, won=True))
    return frames


def _episodes(n):
    return [_won_episode() for _ in range(n)]


def test_a_short_training_run_writes_generations_and_a_manifest(tmp_path):
    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                    supervisor=supervisor)
            model.learn(total_timesteps=16, callback=cb)
        finally:
            env.close()

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2]
    assert gens[-1]["timestep"] == 16
    for g in gens:
        assert g["recoveries"] == 0
        assert g["episodes"] >= 1  # 6-step episodes finish inside each 8-step window
        assert g["win_rate"] == 1.0  # enrichment read won=True from raw infos
        assert g["mean_boss_damage"] == 1.0
    gen, weights, vecnorm = latest_checkpoint(tmp_path)
    assert gen == 2 and weights.exists() and vecnorm.exists()
    assert list(tmp_path.glob("monitor_*")) != []  # VecMonitor session file


def test_build_env_monitor_filters_reset_pending_episodes(tmp_path):
    """The monitor layer must be the reset-aware subclass, or isolated-mode
    throwaway episodes land in the CSV, the dashboard, and ep_rew_mean."""
    with FakeGame(_episodes(2)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            assert isinstance(env.venv, RealEpisodeVecMonitor)
        finally:
            env.close()


def test_build_env_normalizer_is_reset_aware_fresh_and_resumed(tmp_path):
    """Fresh and resumed stacks must both normalize through the
    reset-aware subclass, or placeholder frames pollute obs/return stats."""
    with FakeGame(_episodes(2)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            assert isinstance(env, RealEpisodeVecNormalize)
            env.save(str(tmp_path / "vecnormalize.pkl"))
        finally:
            env.close()
    with FakeGame(_episodes(2)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path,
                                 resume_vecnorm=tmp_path / "vecnormalize.pkl")
        try:
            assert isinstance(env, RealEpisodeVecNormalize)
        finally:
            env.close()


def test_build_model_masks_placeholder_transitions(tmp_path):
    """Fresh and resumed models must both be the masked RecurrentPPO, or
    isolated-mode placeholder steps get trained on like real fights."""
    with FakeGame(_episodes(2)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            assert isinstance(model, MaskedRecurrentPPO)
            saved = tmp_path / "model.zip"
            model.save(saved)
            resumed = train.build_model(env, tmp_path, resume_model=saved)
            assert isinstance(resumed, MaskedRecurrentPPO)
            assert isinstance(resumed.rollout_buffer,
                              MaskedRecurrentRolloutBuffer)
        finally:
            env.close()


def test_build_model_target_kl_fresh_and_resume_override(tmp_path):
    """--target-kl must reach a fresh model AND override a resumed one:
    SB3's load() restores every hyperparameter from the checkpoint zip, so
    without an explicit post-load override the flag would be silently
    ignored on exactly the runs it exists to fix."""
    with FakeGame(_episodes(2)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8,
                                      target_kl=0.05)
            assert model.target_kl == 0.05
            saved = tmp_path / "model.zip"
            model.save(saved)
            # Flag unset: the checkpoint's own value survives the resume.
            resumed = train.build_model(env, tmp_path, resume_model=saved)
            assert resumed.target_kl == 0.05
            # Flag set: it wins over the checkpoint.
            overridden = train.build_model(env, tmp_path, resume_model=saved,
                                           target_kl=0.02)
            assert overridden.target_kl == 0.02
        finally:
            env.close()


def test_two_instance_training_collects_from_both_games(tmp_path):
    """--instances N end to end at N=2 (minus the real processes): two
    bridges feed one vectorized PPO through build_env, and the rollout
    stripes across the whole fleet rather than draining one game."""
    with FakeGame(_episodes(40)) as a, FakeGame(_episodes(40)) as b:
        env, supervisor = train.build_env([a.port, b.port],
                                          relaunch=lambda s: None,
                                          run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=16,
                                    supervisor=supervisor)
            # n_steps is per env, so one rollout is 16 timesteps; 32 makes
            # two full rollouts and two generations.
            model.learn(total_timesteps=32, callback=cb)
        finally:
            env.close()
        # Scripted episodes are consumed per instance (FakeGame pops them),
        # so both shrinking proves both games actually served the rollout.
        assert len(a.episodes) < 40
        assert len(b.episodes) < 40

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2]
    assert gens[-1]["timestep"] == 32
    for g in gens:
        # The manifest aggregates across the fleet: every scripted episode
        # is a win at full boss damage on both instances.
        assert g["episodes"] >= 2
        assert g["win_rate"] == 1.0
        assert g["mean_boss_damage"] == 1.0


def test_reset_log_dir_records_spans_through_the_subprocess_workers(tmp_path):
    """Phase 0 measurement end to end: reset_log_dir set on build_env reaches
    HKEnv inside the SubprocVecEnv workers, and their reset spans survive the
    round-trip back to the run dir's sidecars."""
    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path,
                                          reset_log_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            model.learn(total_timesteps=16)
        finally:
            env.close()
    spans = read_reset_spans(tmp_path)
    assert spans  # 6-step episodes auto-reset inside a 16-step run
    assert all(s >= 0.0 for s in spans)


def test_default_n_steps_keeps_the_total_batch_constant():
    # 2048 total whatever the fleet size: the PPO update runs while every
    # game connection idles, and the mod severs connections idle for 10s,
    # so the update's wall-clock time must not grow with --instances.
    assert train.default_n_steps(1) == 2048
    assert train.default_n_steps(2) == 1024
    assert train.default_n_steps(4) == 512
    assert train.default_n_steps(1000) == 128  # floored, never zero


def test_default_n_steps_compensates_for_masked_placeholder_rows():
    # With async resets on, ~19% of collected rows are reset placeholders
    # that the gradient mask (hkrl/masking.py) drops from the loss, so a
    # 2048-row batch trains on only ~1660 real samples. Inflate the rollout
    # so REAL samples per update stay ~2048: 2048 / (1 - 0.19) / instances.
    assert train.default_n_steps(2, async_resets=True) == 1264
    assert train.default_n_steps(4, async_resets=True) == 632
    # Async resets off (or forced off): the plain division, unchanged.
    assert train.default_n_steps(2, async_resets=False) == 1024
    # The floor holds regardless.
    assert train.default_n_steps(1000, async_resets=True) == 128


def test_session_banner_fresh_states_budget_and_target():
    # Fresh run starts at timestep 0, so target == this session's budget.
    banner = train.session_banner(500_000)
    assert "500,000" in banner
    assert "target timestep 500,000" in banner


def test_session_banner_resumed_states_generation_current_additional_and_target():
    # target = start + budget, matching the dashboard's additive --timesteps
    # framing on resume.
    banner = train.session_banner(500_000, start_timestep=1_200_000,
                                  resumed_gen=8)
    assert "generation 8" in banner
    assert "1,200,000" in banner  # current (start) timestep
    assert "500,000 more" in banner  # this session's additional budget
    assert "target timestep 1,700,000" in banner


def test_stop_flag_ends_training_at_the_current_episodes_end(tmp_path):
    """A stop request finishes the attempt in progress rather than cutting
    the fight off mid-swing: stopping at the episode boundary also leaves
    the game in a state the next session's reset handles cheaply, instead
    of a mid-fight truncation the reset macro has to unwind."""
    flag = threading.Event()
    flag.set()
    with FakeGame(_episodes(5)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            model.learn(total_timesteps=16, callback=train.StopOnFlag(flag))
        finally:
            env.close()
    # The flag was set before the first step, so training ran exactly one
    # scripted 6-step episode -- not zero steps, and not the full rollout.
    assert model.num_timesteps == 6


def test_resume_continues_timesteps_norm_stats_and_generation_numbering(tmp_path):
    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path)
        model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
        cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                supervisor=supervisor)
        model.learn(total_timesteps=16, callback=cb)
        saved_count = env.obs_rms.count
        env.close()

    gen, weights, vecnorm = latest_checkpoint(tmp_path)
    assert gen == 2

    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path, resume_vecnorm=vecnorm)
        try:
            # The statistics were loaded, not freshly initialized: a fresh
            # VecNormalize starts its count at epsilon (1e-4).
            assert env.obs_rms.count == pytest.approx(saved_count)
            model = train.build_model(env, tmp_path, resume_model=weights)
            assert model.num_timesteps == 16  # resumed, not restarted
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                    supervisor=supervisor)
            model.learn(total_timesteps=8, callback=cb,
                        reset_num_timesteps=False)
            assert model.num_timesteps == 24
        finally:
            env.close()

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2, 3]
    assert gens[-1]["timestep"] == 24


def test_confirm_ready_auto_skips_the_prompt(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input",
                        lambda *a: pytest.fail("input() called in auto mode"))
    train.confirm_ready(auto=True, boss_display="Hornet")
    assert "skipping the ready prompt" in capsys.readouterr().out


def test_confirm_ready_interactive_waits_on_input(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))
    train.confirm_ready(auto=False, boss_display="Hornet")
    assert prompts and "Hall of Gods" in prompts[0]


def test_build_apps_clones_even_at_n1(monkeypatch):
    monkeypatch.setattr(train, "SAVE_ISOLATION_SUPPORTED", True)
    monkeypatch.setattr(train, "prepare_instance",
                        lambda port, app, root, slot: f"clone-{port}")
    apps = train.build_apps(ports=[9020], app="master.app",
                            instances_root="/tmp/instances")
    assert apps == ["clone-9020"]


def test_build_apps_none_when_isolation_unsupported(monkeypatch):
    monkeypatch.setattr(train, "SAVE_ISOLATION_SUPPORTED", False)
    apps = train.build_apps(ports=[9020], app="master.app",
                            instances_root="/tmp/instances")
    assert apps is None


def test_async_resets_defaults_on_for_multi_instance_and_off_for_single():
    assert train.resolve_async_resets(None, instances=2) is True
    assert train.resolve_async_resets(None, instances=1) is False
    assert train.resolve_async_resets(False, instances=2) is False  # opt-out
    assert train.resolve_async_resets(True, instances=1) is False   # no sibling


def test_build_config_dict_records_resolved_async_resets():
    """The config dict records the resolved async_resets boolean, not the
    raw tri-state flag, so config.jsonl reflects what actually ran."""
    import argparse

    # Build a minimal args object
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-resets", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100)

    # Case 1: default (None) at N=2 resolves to True
    args = parser.parse_args(["--instances", "2"])
    config = train.build_config_dict(args, async_resets=True, started_at="2026-07-22T12:00:00")
    assert config["async_resets"] is True

    # Case 2: default (None) at N=1 resolves to False
    args = parser.parse_args(["--instances", "1"])
    config = train.build_config_dict(args, async_resets=False, started_at="2026-07-22T12:00:00")
    assert config["async_resets"] is False

    # Case 3: explicit --no-async-resets at N=2 records False
    args = parser.parse_args(["--instances", "2", "--no-async-resets"])
    config = train.build_config_dict(args, async_resets=False, started_at="2026-07-22T12:00:00")
    assert config["async_resets"] is False

    # Case 4: explicit --async-resets at N=2 records True
    args = parser.parse_args(["--instances", "2", "--async-resets"])
    config = train.build_config_dict(args, async_resets=True, started_at="2026-07-22T12:00:00")
    assert config["async_resets"] is True


def test_async_resets_trains_end_to_end_and_still_serves_both_games(tmp_path):
    """--async-resets end to end at N=2 (minus the real processes): the
    kwargs reach make_env inside the SubprocVecEnv workers, and the wrapper
    is demonstrably ACTIVE there -- after every death, the first step's info
    carries the reset_pending key (True while pending, False on the splice),
    which only AsyncResetWrapper produces. Placeholders and splices flow
    through VecMonitor/VecNormalize, and training completes."""
    from stable_baselines3.common.callbacks import BaseCallback

    class SpotsPending(BaseCallback):
        def __init__(self):
            super().__init__()
            self.pending_infos = 0

        def _on_step(self) -> bool:
            self.pending_infos += sum(
                1 for i in self.locals.get("infos", ()) if "reset_pending" in i)
            return True

    spots = SpotsPending()
    with FakeGame(_episodes(40)) as a, FakeGame(_episodes(40)) as b:
        env, supervisor = train.build_env([a.port, b.port],
                                          relaunch=lambda s: None,
                                          run_dir=tmp_path,
                                          async_resets=True,
                                          pending_mode="prefix")
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            model.learn(total_timesteps=32, callback=spots)
        finally:
            env.close()
        assert len(a.episodes) < 40
        assert len(b.episodes) < 40
    # The wrapper really ran inside the workers: without it no step info
    # ever carries the reset_pending key.
    assert spots.pending_infos >= 1


def test_supervisor_recovery_composes_with_async_resets(tmp_path):
    """The design's risk section, exercised end to end: at N=2 with async
    resets ACTIVE, one instance's connection drops mid-run.
    SupervisedVecEnv._recover force-closes the whole vec (SubprocVecEnv can't
    isolate which slot failed) and rebuilds it from scratch -- fresh worker
    subprocesses, so fresh AsyncResetWrapper instances too. The worry is that
    the rebuilt vec might come up with inherited pending state instead of a
    clean slate. Proof that it doesn't: training survives the death, reaches
    its full timestep budget, and a generation whose ENTIRE window falls
    after the recovery still records real wins -- a vec stuck emitting
    placeholders (e.g. a wrapper that carried a stale background thread
    across the rebuild) would never produce another episode record."""
    from stable_baselines3.common.callbacks import BaseCallback

    class KillFirstInstanceOnce(BaseCallback):
        """Tears down FakeGame `game`'s live connection once, after
        `after_steps` combined vec steps -- the mid-run death this test
        simulates without any sleep-based timing."""

        def __init__(self, game, after_steps):
            super().__init__()
            self._game = game
            self._after_steps = after_steps
            self._steps = 0
            self.killed = False

        def _on_step(self) -> bool:
            self._steps += 1
            if not self.killed and self._steps >= self._after_steps:
                self.killed = True
                self._game.__exit__(None, None, None)
            return True

    class SpotsPending(BaseCallback):
        def __init__(self):
            super().__init__()
            self.pending_infos = 0

        def _on_step(self) -> bool:
            self.pending_infos += sum(
                1 for i in self.locals.get("infos", ()) if "reset_pending" in i)
            return True

    with FakeGame(_episodes(40)) as a, FakeGame(_episodes(40)) as b:
        spawned = []

        def relaunch(slot):
            assert slot == 0  # instance 0 is the one this test kills
            spawned.append(FakeGame(_episodes(40), port=a.port).__enter__())

        env, supervisor = train.build_env(
            [a.port, b.port], relaunch=relaunch, run_dir=tmp_path,
            async_resets=True, pending_mode="prefix",
            # Bounded recovery timing, same shape as test_supervisor.py's
            # FAST: a real recovery must stay well inside this test's runtime.
            recover_attempts=2, recover_delay=0.0, probe_timeout=0.3,
            launch_timeout=1.0, ready_timeout=1.0, timeout=0.5)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            gen_cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=16,
                                        supervisor=supervisor)
            # after_steps=2 kills instance 0 a couple of combined steps into
            # the first of three 16-timestep rollouts, so recovery lands well
            # inside generation 1's window and generations 2-3 fall entirely
            # after it.
            killer = KillFirstInstanceOnce(a, after_steps=2)
            spots = SpotsPending()
            model.learn(total_timesteps=48, callback=[killer, gen_cb, spots])
        finally:
            env.close()
            for fg in spawned:
                fg.__exit__(None, None, None)

    assert model.num_timesteps == 48  # the budget was reached despite the death
    assert supervisor.recoveries >= 1
    assert len(spawned) == 1  # exactly one relaunch, on instance 0's own port
    # The wrapper really ran in the workers, both before and after the rebuild.
    assert spots.pending_infos >= 1

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2, 3]
    assert gens[-1]["timestep"] == 48
    # Generation 3's whole window (timesteps 33-48) comes after the recovery,
    # which lands inside generation 1's -- real wins there prove the rebuilt
    # vec kept serving genuine episodes rather than getting stuck on
    # placeholders forever.
    assert gens[-1]["episodes"] >= 1
    assert gens[-1]["win_rate"] > 0.0


def test_resolve_boss_fresh_run_defaults_to_hornet1():
    assert train.resolve_boss(None, None) == "hornet1"
    assert train.resolve_boss("gruz_mother", None) == "gruz_mother"


def test_resolve_boss_resume_reads_the_recorded_boss(tmp_path):
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother"}) + "\n")
    assert train.resolve_boss(None, tmp_path) == "gruz_mother"


def test_resolve_boss_resume_without_a_recorded_boss_is_hornet1(tmp_path):
    # Runs from before the boss field existed.
    (tmp_path / "config.jsonl").write_text(json.dumps({"instances": 1}) + "\n")
    assert train.resolve_boss(None, tmp_path) == "hornet1"


def test_resolve_boss_refuses_a_conflicting_flag_on_resume(tmp_path):
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "hornet1"}) + "\n")
    with pytest.raises(ValueError, match="gruz_mother"):
        train.resolve_boss("gruz_mother", tmp_path)


def test_resolve_boss_rejects_a_recorded_boss_this_registry_lacks(tmp_path):
    # A run recorded against a boss this build doesn't know must fail at
    # the guard, not deep in worker env construction.
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "no_such_boss"}) + "\n")
    with pytest.raises(ValueError, match="no_such_boss"):
        train.resolve_boss(None, tmp_path)


def test_confirm_ready_prompt_names_the_boss(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text))
    train.confirm_ready(False, "Gruz Mother")
    assert "Gruz Mother statue" in prompts[0]
    assert "Hornet" not in prompts[0]


def test_build_prepares_reseeds_that_ports_clone_save(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "SAVE_ISOLATION_SUPPORTED", True)
    monkeypatch.setattr(train, "seed_save_dir",
                        lambda bundle_id: calls.append(("seed", bundle_id))
                        or pathlib.Path("/sandbox/saves.hkrl9021"))
    monkeypatch.setattr(train, "prepare_clone_save",
                        lambda save_dir: calls.append(("prep", save_dir)))
    prepares = train.build_prepares([9020, 9021])
    assert len(prepares) == 2 and calls == []   # lazy: nothing runs at build
    prepares[1]()
    assert calls == [
        ("seed", f"{train.MASTER_BUNDLE_ID}.hkrl9021"),
        ("prep", pathlib.Path("/sandbox/saves.hkrl9021")),
    ]


def test_build_prepares_is_none_without_save_isolation(monkeypatch):
    monkeypatch.setattr(train, "SAVE_ISOLATION_SUPPORTED", False)
    assert train.build_prepares([9020]) is None


def test_parse_session_args_tracks_which_flags_were_typed():
    args, explicit = train.parse_session_args([])
    assert explicit == set()
    assert args.instances == 1 and args.timesteps == 500_000

    args, explicit = train.parse_session_args(
        ["--instances", "2", "--timesteps", "40000"])
    assert explicit == {"instances", "timesteps"}
    assert args.instances == 2 and args.timesteps == 40_000


def test_parse_session_args_sees_boolean_and_store_true_flags():
    # BooleanOptionalAction (both spellings) and store_true must register
    # as typed, or inheritance could silently override an explicit choice.
    _, explicit = train.parse_session_args(["--no-async-resets"])
    assert "async_resets" in explicit
    _, explicit = train.parse_session_args(["--async-resets"])
    assert "async_resets" in explicit
    _, explicit = train.parse_session_args(["--auto"])
    assert "auto" in explicit
