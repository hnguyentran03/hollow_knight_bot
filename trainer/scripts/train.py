#!/usr/bin/env python3
"""Train a recurrent PPO against Hornet 1 on N supervised game instances.

Owns the game processes end to end: launches them (one per port, counting
up from --port), supervises them through crashes, wedges, and App Nap
suspensions, checkpoints a generation every --gen-every timesteps, and
shuts them down on exit. Stop a run with one
Ctrl-C: it finishes the current rollout, saves a final generation, and
terminates the game. launch_instances.py stays for manual gates only -- the
training game must be this process's own child or relaunch() could never
reap a wedged game's port.
"""
import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from hkrl.game import GameFleet  # noqa: E402
from hkrl.generations import GenerationCallback, latest_checkpoint  # noqa: E402
from hkrl.masking import MaskedRecurrentPPO  # noqa: E402
from hkrl.supervisor import InstanceDown, SupervisedVecEnv  # noqa: E402
from hkrl.vec import RealEpisodeVecMonitor, RealEpisodeVecNormalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import (  # noqa: E402
    DEFAULT_APP, DEFAULT_PORT, SAVE_ISOLATION_SUPPORTED, backup_saves,
    prepare_instance,
)

GAMMA = 0.995


def default_n_steps(instances: int) -> int:
    """Per-instance rollout length for --n-steps when not given explicitly.

    Divides so the total batch per update -- and with it the update's
    wall-clock time, which must stay inside the mod's 10s idle-disconnect
    ceiling -- holds at ~2048 whatever the fleet size. Floored at 128 so an
    absurd fleet still collects a usable sequence per instance.
    """
    return max(128, 2048 // instances)


def resolve_async_resets(flag, instances: int) -> bool:
    """Async resets are a multi-instance throughput feature: on by default
    at N>=2 since the Phase 2 gate passed, always off at N=1 (no sibling to
    freeze), and --no-async-resets is the escape hatch."""
    if instances < 2:
        return False
    return True if flag is None else bool(flag)


def build_config_dict(args, async_resets, resume=None, started_at=None):
    """Build the config dict to be written to config.jsonl, recording the
    resolved async_resets value (not the raw tri-state flag)."""
    if started_at is None:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        **{k: str(v) if isinstance(v, Path) else v
           for k, v in vars(args).items()},
        # No "n_stack": the recurrent policy replaced frame stacking, so
        # there is no stack depth to record.
        "async_resets": async_resets,  # Override with resolved boolean
        "gamma": GAMMA, "ent_coef": 0.01,
        "resumed_from_gen": resume[0] if resume else None,
        "started_at": started_at,
    }


def session_banner(timesteps: int, start_timestep: int = 0,
                   resumed_gen: int | None = None) -> str:
    """One line stating this session's budget in the dashboard's language:
    current timestep and the target it runs to (--timesteps is additive
    on resume, so the target is start + budget)."""
    target = start_timestep + timesteps
    if resumed_gen is None:
        return (f"this session: collecting {timesteps:,} steps "
                f"(target timestep {target:,})")
    return (f"resumed from generation {resumed_gen} at timestep "
            f"{start_timestep:,}; collecting {timesteps:,} more "
            f"(target timestep {target:,})")


def build_env(ports, relaunch, run_dir, resume_vecnorm=None, **supervisor_kwargs):
    """SupervisedVecEnv -> RealEpisodeVecMonitor -> RealEpisodeVecNormalize.

    Returns (env, supervisor): the outermost wrapper for PPO, plus the
    supervisor itself so the checkpoint callback can read its recovery
    count.

    The monitor is RealEpisodeVecMonitor so isolated-mode async-reset
    throwaway episodes never reach the monitor CSV, the dashboard, or
    ep_rew_mean. It sits below the normalizer so its episode records carry
    raw rewards and true lengths. It gets no info_keywords: VecMonitor indexes
    every keyword into each done step's info, and the supervisor's recovery
    frames carry only terminal_observation, so a keyword would KeyError there
    and kill the run on its first recovery. GenerationCallback reads
    won/boss_damage_frac from the raw infos instead.

    No frame stacking: one observation is an instant, and the FSM one-hot
    does not encode how long Hornet has been in a state -- but the recurrent
    ("MlpLstmPolicy") policy carries its own hidden state across steps, so
    the LSTM supplies exactly the temporal memory a VecFrameStack used to
    fake. Stacking on top would only feed the LSTM redundant, delayed copies
    of frames it already remembers.

    The normalizer is RealEpisodeVecNormalize so placeholder frames never
    update the obs/return running statistics. It shares PPO's gamma so its
    return normalization tracks the same discounted quantity the value head
    learns.
    """
    supervisor = SupervisedVecEnv(list(ports), relaunch=relaunch, **supervisor_kwargs)
    session = time.strftime("%Y%m%d_%H%M%S")
    # Session-stamped so a resumed run appends a new episode log instead of
    # truncating the previous session's.
    mon = RealEpisodeVecMonitor(
        supervisor, filename=str(Path(run_dir) / f"monitor_{session}"))
    if resume_vecnorm is not None:
        # The saved statistics are the distribution the resumed weights were
        # trained under; loading them together is what makes a resume a
        # continuation. training stays on so they keep adapting.
        env = RealEpisodeVecNormalize.load(str(resume_vecnorm), mon)
        env.training = True
    else:
        env = RealEpisodeVecNormalize(mon, gamma=GAMMA, clip_obs=10.0)
    return env, supervisor


def build_model(env, run_dir, resume_model=None, seed=None,
                n_steps=2048, batch_size=64, n_epochs=10):
    """A RecurrentPPO for this env, fresh or loaded from a generation
    checkpoint.

    The masked subclass so async-reset placeholder transitions never reach
    the gradient (hkrl/masking.py); loading an old plain-RecurrentPPO
    checkpoint through it is fine, the weights are identical.

    On resume every hyperparameter comes from the checkpoint zip; the
    keyword arguments here shape fresh models only.
    """
    if resume_model is not None:
        return MaskedRecurrentPPO.load(str(resume_model), env=env,
                                       device="cpu")
    return MaskedRecurrentPPO(
        "MlpLstmPolicy",
        env,
        # ~13s credit horizon at 15 Hz, so a dodge can still be credited
        # with the punish it enables; the default 0.99 covers only ~6.7s.
        gamma=GAMMA,
        learning_rate=3e-4,
        # Per-instance rollout length; the flag's default divides 2048 by
        # --instances so the total batch per update stays ~2048 (~2.3
        # minutes of play at 15 Hz) however many games collect it.
        n_steps=n_steps,
        batch_size=batch_size,
        # The whole gradient update runs while the games play on in real
        # time -- the keepalive pinger (hkrl/protocol.py) keeps the
        # connections alive through it, but every Knight stands in a live
        # fight for the duration, so the update should still finish in a
        # few seconds. This bites HARDER
        # with the recurrent 256x256+LSTM policy than the old tiny MLP:
        # measured on CPU at n_steps=2048, n_epochs=10 takes ~13s (over the
        # ceiling), n_epochs=6 ~7.8s, n_epochs=5 ~6.7s -- update time scales
        # ~linearly with n_epochs, and a bigger batch_size does NOT help (the
        # per-sequence LSTM forward dominates). Hence the default dropped from
        # 10 to 5; if updates still approach 10s, lower --n-epochs further
        # before touching anything else.
        n_epochs=n_epochs,
        # Discrete 15-action combat collapses quickly into "hold nothing"
        # under a death-dominated reward; a small entropy bonus keeps
        # alternatives alive long enough for the damage term to be
        # discovered.
        ent_coef=0.01,
        # The MLP feature extractors around the LSTM. Bumped from SB3's
        # default 64x64 to 256x256 (separate pi/vf stacks): the 46-dim
        # observation plus a shared LSTM feed a policy that has to tell apart
        # 21 discrete combat actions against a fast boss, and the wider heads
        # give it the capacity to. The LSTM hidden size is left at
        # RecurrentPPO's default -- it is the memory, not the per-step
        # capacity, this net_arch controls.
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=seed,
        verbose=1,
        # The policy is a small LSTM + MLP; CPU avoids the per-batch device
        # transfer overhead that would dominate on MPS, and recurrent
        # rollouts run one step at a time (no big batched forward pass to
        # amortize a transfer over).
        device="cpu",
        tensorboard_log=str(Path(run_dir) / "tb"),
    )


class StopOnFlag(BaseCallback):
    """Ends learn() cleanly when the flag is set: collection continues to the
    end of the episode in progress, then learn() returns, so the final
    checkpoint save and game shutdown run on the normal path instead of
    unwinding through a KeyboardInterrupt raised inside a socket read or a
    recovery.

    The episode boundary, not the next step: cutting the fight off mid-swing
    leaves the game mid-fight, where the next session's first reset has to
    unwind a live Hornet through the truncation path -- the slowest,
    budget-hungriest branch of the reset macro. Waiting for done costs at
    most one episode (~3 minutes at the env's max_steps ceiling, usually far
    less), and a second Ctrl-C still forces an immediate abort via
    request_stop's KeyboardInterrupt."""

    def __init__(self, flag: threading.Event):
        super().__init__()
        self.flag = flag

    def _on_step(self) -> bool:
        if not self.flag.is_set():
            return True
        # collect_rollouts publishes its locals (including this step's
        # `dones`) via update_locals before each on_step call.
        return not any(self.locals["dones"])


def confirm_ready(auto: bool) -> None:
    """Gate between "games are up" and "training begins".

    Interactive runs wait for a human to confirm the Hall of Gods.
    --auto (dashboard-launched: no terminal, nobody to press Enter) skips
    straight in and lets the boot macro drive the game there, at the cost
    of a few reset retries.
    """
    if auto:
        print("--auto: skipping the ready prompt; the boot macro will "
              "drive the game(s) into the Hall of Gods", flush=True)
        return
    input("Bring the game(s) to the Hall of Gods near the Hornet statue, "
          "then press Enter. (A freshly booted game can also challenge "
          "itself in via the boot macro; expect a few reset retries.) ")


def build_apps(ports, app, instances_root):
    """Per-port clone binaries for the fleet, or None when unsupported.

    Every instance -- N=1 included -- runs on its own per-port APFS clone so
    the clone-save prep (Godhome-only profile) applies uniformly and no game
    ever plays the master save directly. On platforms without save isolation
    there is no clone: returns None and the caller falls back to the master app.
    """
    if not SAVE_ISOLATION_SUPPORTED:
        print("WARNING: save isolation is not implemented on this "
              "platform; the game will play the master save slot directly "
              "and concurrent autosaves can corrupt it. Back up the save "
              "directory first.", file=sys.stderr, flush=True)
        return None
    print(f"preparing {len(ports)} isolated game copies under "
          f"{instances_root} (APFS clones; instant, near-zero disk)",
          flush=True)
    return [prepare_instance(p, app, instances_root, slot=i)
            for i, p in enumerate(ports)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=500_000,
                    help="env steps to collect this session (~54k/hour at "
                         "15 Hz; adds onto a resumed run's count)")
    ap.add_argument("--run-id", default=None,
                    help="name for a NEW run under <root>/runs/ "
                         "(default: timestamp)")
    ap.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR",
                    help="continue an existing run from its latest generation")
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser())
    ap.add_argument("--app", type=Path, default=DEFAULT_APP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="first bridge port; instance i listens on port+i")
    ap.add_argument("--instances", type=int, default=1,
                    help="game instances to run in parallel. Every instance "
                         "is a full game client -- 2-3 is realistic on one "
                         "machine, and every window must stay visible")
    ap.add_argument("--gen-every", type=int, default=15_000)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="PPO rollout length PER INSTANCE (default: "
                         "2048 // instances). The default divides so the "
                         "total batch -- and with it the update's wall-clock "
                         "time -- stays constant as --instances grows: the "
                         "games run on in real time while the update "
                         "computes, every Knight standing in a live fight")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=5,
                    help="PPO epochs per update. Kept at 5 so the recurrent "
                         "256x256+LSTM update stays short (~6.7s on CPU): "
                         "the keepalive pinger keeps connections alive "
                         "through longer updates, but the Knights stand in "
                         "their live fights for the whole update. Raise "
                         "only if the net moves off CPU.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--auto", action="store_true",
                    help="skip the interactive ready prompt (unattended/"
                         "dashboard launches); the boot macro drives the "
                         "game into the Hall of Gods")
    ap.add_argument("--measure-resets", action="store_true",
                    help="Phase 0 async-resets measurement: log every reset's "
                         "wall-clock span to resets_<port>.jsonl under the run "
                         "dir. Analyze with scripts/measure_reset_freeze.py. "
                         "Off by default; a normal run pays nothing.")
    ap.add_argument("--async-resets", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="background-thread resets with placeholder steps so "
                         "one instance's reset never freezes its siblings. "
                         "Default: on at --instances >= 2 (Phase 2 gate "
                         "passed 2026-07-22 -- see the async-resets design doc), off at "
                         "1. --no-async-resets forces the old synchronous "
                         "behavior.")
    ap.add_argument("--async-reset-mode", choices=("prefix", "isolated"),
                    default="isolated",
                    help="what the pending window is to PPO: a prefix of the "
                         "next episode (LSTM state carries across the splice) "
                         "or an isolated throwaway episode (LSTM state resets "
                         "at the fight's first real frame)")
    args = ap.parse_args()
    if args.instances < 1:
        sys.exit("--instances must be at least 1")
    # Note: only warn about explicit --async-resets at N=1; default (None) is handled by resolve_async_resets
    if args.async_resets is True and args.instances < 2:
        print("hkrl: --async-resets is a no-op at --instances 1 (no sibling "
              "to freeze); running synchronously", file=sys.stderr, flush=True)
    # Resolved before the config dump below so config.jsonl records the
    # value actually used, not None.
    if args.n_steps is None:
        args.n_steps = default_n_steps(args.instances)
    async_resets = resolve_async_resets(args.async_resets, args.instances)

    if args.resume is not None:
        run_dir = args.resume.expanduser()
        resume = latest_checkpoint(run_dir)  # (gen, weights, vecnorm)
    else:
        resume = None
        run_dir = args.root / "runs" / (args.run_id or time.strftime("%Y%m%d_%H%M%S"))
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            sys.exit(f"{run_dir} already exists. Restarting into an existing "
                     f"run is never implicit: pass --resume {run_dir} to "
                     f"continue it, or a different --run-id to start fresh.")

    # One JSON object per session, appended, so a resumed run's full history
    # stays inspectable next to its checkpoints.
    with (run_dir / "config.jsonl").open("a") as f:
        config = build_config_dict(args, async_resets, resume=resume)
        f.write(json.dumps(config) + "\n")

    # Unconditional: every clone is seeded FROM the master save (N=1 on
    # save-isolation platforms, the master itself on others), so a quietly
    # corrupt master would propagate. A run must never be the event that
    # loses someone's save.
    backup = backup_saves(args.root)
    if backup is not None:
        print(f"master save backed up to {backup}", flush=True)
    if resume is None:
        print(session_banner(args.timesteps), flush=True)

    ports = [args.port + i for i in range(args.instances)]
    # Instances must not share the game's save directory: they all autosave
    # the same slot throughout a run (observed corrupting the master save
    # live, 2026-07-20 -- see seed_save_dir). Each slot gets its own app
    # clone with a per-port bundle id (own save dir, own ModLog), refreshed
    # from the master app and save at every start -- see build_apps.
    apps = build_apps(ports, args.app, args.root / "instances")
    game = GameFleet(ports, app=args.app, apps=apps)
    env = None
    exit_code = 0
    try:
        game.start()
        print(f"game(s) up on port(s) {', '.join(map(str, game.ports))}",
              flush=True)
        if sys.platform == "win32":
            print("Keep the game window visible (not minimized) for the whole "
                  "run, and disable system sleep for its duration "
                  "(Settings > System > Power, or `powercfg /change "
                  "standby-timeout-ac 0`): a suspended game holds its port "
                  "open while wedged, and every occurrence costs a full "
                  "relaunch-and-reboot recovery.",
                  flush=True)
        else:
            print("Keep the game window visible for the whole run: macOS "
                  "suspends a fully occluded window (App Nap) with its port "
                  "still open, and every occurrence costs a full "
                  "relaunch-and-reboot recovery. Suppress display sleep for "
                  "the run: caffeinate -d (in another terminal).",
                  flush=True)
        confirm_ready(args.auto)

        stop = threading.Event()

        def request_stop(signum, frame):
            if stop.is_set():
                raise KeyboardInterrupt  # second Ctrl-C: abandon the clean path
            stop.set()
            print("stop requested: finishing the current episode, saving a final "
                  "generation, then shutting the game down (Ctrl-C again to "
                  "force)", file=sys.stderr, flush=True)

        signal.signal(signal.SIGINT, request_stop)

        env, supervisor = build_env(
            game.ports, game.relaunch, run_dir,
            resume_vecnorm=resume[2] if resume else None,
            # Boot-to-fight spans several 22.5s reset budgets, so a single
            # relaunch legitimately consumes a few attempts (see the
            # boot-retry note in hkrl/supervisor.py); the default 3 would
            # abandon a boot that was converging.
            recover_attempts=8,
            # Phase 0 async-resets measurement, only when asked. Flows through
            # the supervisor's env_kwargs to every worker's HKEnv.
            **({"reset_log_dir": run_dir} if args.measure_resets else {}),
            # Async resets: multi-instance only. Flows through env_kwargs to
            # make_env inside every worker, exactly like reset_log_dir.
            **({"async_resets": True, "pending_mode": args.async_reset_mode}
               if async_resets else {}),
        )
        model = build_model(env, run_dir,
                            resume_model=resume[1] if resume else None,
                            seed=args.seed, n_steps=args.n_steps,
                            batch_size=args.batch_size, n_epochs=args.n_epochs)
        if resume:
            print(f"{run_dir}: " + session_banner(
                args.timesteps, start_timestep=model.num_timesteps,
                resumed_gen=resume[0]), flush=True)
        callback = GenerationCallback(run_dir, vecnorm=env,
                                      every_steps=args.gen_every,
                                      supervisor=supervisor)
        try:
            model.learn(total_timesteps=args.timesteps,
                        callback=[callback, StopOnFlag(stop)],
                        reset_num_timesteps=resume is None)
        except InstanceDown as exc:
            print(f"!!! instance recovery exhausted: {exc}",
                  file=sys.stderr, flush=True)
            print(f"!!! this session ends here; continue it with:\n"
                  f"!!!   ./.venv/bin/python scripts/train.py --resume {run_dir}\n"
                  f"!!!   (or the Resume button on the dashboard's /summon page)",
                  file=sys.stderr, flush=True)
            exit_code = 1
        except KeyboardInterrupt:
            print("forced interrupt; attempting a final save",
                  file=sys.stderr, flush=True)
            exit_code = 130
        # Not callback.model: in the installed SB3, BaseCallback.model is a
        # class-level annotation only, absent until init_callback() runs, and
        # _setup_learn() resets the env before that -- so InstanceDown (or a
        # second Ctrl-C) during the initial reset lands here with no .model
        # attribute at all, and a plain attribute access would raise
        # AttributeError instead of skipping the save as intended.
        if getattr(callback, "model", None) is not None:
            final = callback.save_generation()
            print(f"final checkpoint: {final}", flush=True)
            print(f"manifest: {run_dir / 'generations.jsonl'}", flush=True)
    finally:
        # Cleanup must cover the pre-handler window: stop() is idempotent,
        # so calling it here ensures the game is reaped even if Ctrl-C
        # was pressed during start(), the startup prints, or the input() prompt.
        if env is not None:
            env.close()
        game.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
