#!/usr/bin/env python3
"""Train PPO against Hornet 1 on one supervised game instance.

Owns the game process end to end: launches it, supervises it through
crashes, wedges, and App Nap suspensions, checkpoints a generation every
--gen-every timesteps, and shuts it down on exit. Stop a run with one
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

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    VecFrameStack, VecMonitor, VecNormalize,
)

from hkrl.game import GameProcess  # noqa: E402
from hkrl.generations import GenerationCallback, latest_checkpoint  # noqa: E402
from hkrl.supervisor import InstanceDown, SupervisedVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import DEFAULT_APP, DEFAULT_PORT  # noqa: E402

GAMMA = 0.995
N_STACK = 4


def build_env(ports, relaunch, run_dir, resume_vecnorm=None, **supervisor_kwargs):
    """SupervisedVecEnv -> VecMonitor -> VecFrameStack -> VecNormalize.

    Returns (env, supervisor): the outermost wrapper for PPO, plus the
    supervisor itself so the checkpoint callback can read its recovery
    count.

    VecMonitor sits below the frame stack and the normalizer so its episode
    records carry raw rewards and true lengths. It gets no info_keywords:
    VecMonitor indexes every keyword into each done step's info, and the
    supervisor's recovery frames carry only terminal_observation, so a
    keyword would KeyError there and kill the run on its first recovery.
    GenerationCallback reads won/boss_damage_frac from the raw infos
    instead.

    VecFrameStack(n_stack=4): one observation is an instant, and the FSM
    one-hot does not encode how long Hornet has been in a state -- the
    stack is what lets the policy tell "starting a dash" from "mid-dash".

    VecNormalize shares PPO's gamma so its return normalization tracks the
    same discounted quantity the value head learns.
    """
    supervisor = SupervisedVecEnv(list(ports), relaunch=relaunch, **supervisor_kwargs)
    session = time.strftime("%Y%m%d_%H%M%S")
    # Session-stamped so a resumed run appends a new episode log instead of
    # truncating the previous session's.
    mon = VecMonitor(supervisor, filename=str(Path(run_dir) / f"monitor_{session}"))
    stacked = VecFrameStack(mon, n_stack=N_STACK)
    if resume_vecnorm is not None:
        # The saved statistics are the distribution the resumed weights were
        # trained under; loading them together is what makes a resume a
        # continuation. training stays on so they keep adapting.
        env = VecNormalize.load(str(resume_vecnorm), stacked)
        env.training = True
    else:
        env = VecNormalize(stacked, gamma=GAMMA, clip_obs=10.0)
    return env, supervisor


def build_model(env, run_dir, resume_model=None, seed=None,
                n_steps=2048, batch_size=64, n_epochs=10):
    """A PPO for this env, fresh or loaded from a generation checkpoint.

    On resume every hyperparameter comes from the checkpoint zip; the
    keyword arguments here shape fresh models only.
    """
    if resume_model is not None:
        return PPO.load(str(resume_model), env=env, device="cpu")
    return PPO(
        "MlpPolicy",
        env,
        # ~13s credit horizon at 15 Hz, so a dodge can still be credited
        # with the punish it enables; the default 0.99 covers only ~6.7s.
        gamma=GAMMA,
        learning_rate=3e-4,
        # One instance at 15 Hz collects the full n_steps=2048 (~2.3 minutes
        # of play) per update.
        n_steps=n_steps,
        batch_size=batch_size,
        # The whole gradient update runs while the game connection idles,
        # and the mod drops any connection idle for 10s (see the ceiling
        # note in hkrl/env.py) -- so the update must finish well inside 10s
        # or every rollout boundary severs the connection. Measured at the
        # live gate; if updates approach 10s, lower --n-epochs first.
        n_epochs=n_epochs,
        # Discrete 15-action combat collapses quickly into "hold nothing"
        # under a death-dominated reward; a small entropy bonus keeps
        # alternatives alive long enough for the damage term to be
        # discovered.
        ent_coef=0.01,
        seed=seed,
        verbose=1,
        # The policy is a tiny MLP; CPU avoids the per-batch device
        # transfer overhead that would dominate on MPS.
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
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--gen-every", type=int, default=15_000)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

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
            "gamma": GAMMA, "n_stack": N_STACK, "ent_coef": 0.01,
            "resumed_from_gen": resume[0] if resume else None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")

    game = GameProcess(port=args.port, app=args.app)
    env = None
    exit_code = 0
    try:
        game.start()
        print(f"game up on port {game.port}", flush=True)
        print("Keep the game window visible for the whole run: macOS suspends a "
              "fully occluded window (App Nap) with its port still open, and "
              "every occurrence costs a full relaunch-and-reboot recovery. "
              "Suppress display sleep for the run: caffeinate -d (in another "
              "terminal).",
              flush=True)
        input("Bring the game to the Hall of Gods near the Hornet statue, then "
              "press Enter. (A freshly booted game can also challenge itself in "
              "via the boot macro; expect a few reset retries.) ")

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
            [game.port], game.relaunch, run_dir,
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
