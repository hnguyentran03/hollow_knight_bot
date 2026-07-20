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

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    VecMonitor, VecNormalize,
)

from hkrl.game import GameFleet  # noqa: E402
from hkrl.generations import GenerationCallback, latest_checkpoint  # noqa: E402
from hkrl.supervisor import InstanceDown, SupervisedVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import (  # noqa: E402
    DEFAULT_APP, DEFAULT_PORT, SAVE_ISOLATION_SUPPORTED, prepare_instance,
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


def build_env(ports, relaunch, run_dir, resume_vecnorm=None, **supervisor_kwargs):
    """SupervisedVecEnv -> VecMonitor -> VecNormalize.

    Returns (env, supervisor): the outermost wrapper for PPO, plus the
    supervisor itself so the checkpoint callback can read its recovery
    count.

    VecMonitor sits below the normalizer so its episode records carry raw
    rewards and true lengths. It gets no info_keywords: VecMonitor indexes
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

    VecNormalize shares PPO's gamma so its return normalization tracks the
    same discounted quantity the value head learns.
    """
    supervisor = SupervisedVecEnv(list(ports), relaunch=relaunch, **supervisor_kwargs)
    session = time.strftime("%Y%m%d_%H%M%S")
    # Session-stamped so a resumed run appends a new episode log instead of
    # truncating the previous session's.
    mon = VecMonitor(supervisor, filename=str(Path(run_dir) / f"monitor_{session}"))
    if resume_vecnorm is not None:
        # The saved statistics are the distribution the resumed weights were
        # trained under; loading them together is what makes a resume a
        # continuation. training stays on so they keep adapting.
        env = VecNormalize.load(str(resume_vecnorm), mon)
        env.training = True
    else:
        env = VecNormalize(mon, gamma=GAMMA, clip_obs=10.0)
    return env, supervisor


def build_model(env, run_dir, resume_model=None, seed=None,
                n_steps=2048, batch_size=64, n_epochs=10):
    """A RecurrentPPO for this env, fresh or loaded from a generation
    checkpoint.

    On resume every hyperparameter comes from the checkpoint zip; the
    keyword arguments here shape fresh models only.
    """
    if resume_model is not None:
        return RecurrentPPO.load(str(resume_model), env=env, device="cpu")
    return RecurrentPPO(
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
    args = ap.parse_args()
    if args.instances < 1:
        sys.exit("--instances must be at least 1")
    # Resolved before the config dump below so config.jsonl records the
    # value actually used, not None.
    if args.n_steps is None:
        args.n_steps = default_n_steps(args.instances)

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
        f.write(json.dumps({
            **{k: str(v) if isinstance(v, Path) else v
               for k, v in vars(args).items()},
            # No "n_stack": the recurrent policy replaced frame stacking, so
            # there is no stack depth to record.
            "gamma": GAMMA, "ent_coef": 0.01,
            "resumed_from_gen": resume[0] if resume else None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")

    ports = [args.port + i for i in range(args.instances)]
    # At N>1 the instances must not share the game's save directory: they
    # all autosave the same slot throughout a run (observed corrupting the
    # master save live, 2026-07-20 -- see seed_save_dir). Each slot gets
    # its own app clone with a per-port bundle id (own save dir, own
    # ModLog), refreshed from the master app and save at every start. N=1
    # keeps the historical behavior: the single game plays the master save
    # directly.
    apps = None
    if args.instances > 1:
        if SAVE_ISOLATION_SUPPORTED:
            print(f"preparing {args.instances} isolated game copies under "
                  f"{args.root / 'instances'} (APFS clones; instant, "
                  f"near-zero disk)", flush=True)
            apps = [prepare_instance(p, args.app, args.root / "instances")
                    for p in ports]
        else:
            print("WARNING: save isolation is not implemented on this "
                  "platform; all instances will share one save slot and "
                  "concurrent autosaves can corrupt it. Back up the save "
                  "directory first.", file=sys.stderr, flush=True)
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
        input("Bring the game(s) to the Hall of Gods near the Hornet statue, "
              "then press Enter. (A freshly booted game can also challenge "
              "itself in via the boot macro; expect a few reset retries.) ")

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
        )
        model = build_model(env, run_dir,
                            resume_model=resume[1] if resume else None,
                            seed=args.seed, n_steps=args.n_steps,
                            batch_size=args.batch_size, n_epochs=args.n_epochs)
        if resume:
            print(f"resumed {run_dir} from generation {resume[0]} at timestep "
                  f"{model.num_timesteps}; collecting {args.timesteps} more",
                  flush=True)
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
                  f"!!!   ./.venv/bin/python scripts/train.py --resume {run_dir}",
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
