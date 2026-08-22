#!/usr/bin/env python3
"""Train a recurrent PPO against a registered boss on N supervised game instances.

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

from hkrl.bosses import BOSSES, DEFAULT_BOSS, get_boss  # noqa: E402
from hkrl.game import GameFleet  # noqa: E402
from hkrl.generations import GenerationCallback, latest_checkpoint  # noqa: E402
from hkrl.masking import MaskedRecurrentPPO  # noqa: E402
from hkrl.rundata import read_jsonl  # noqa: E402
from hkrl.supervisor import InstanceDown, SupervisedVecEnv  # noqa: E402
from hkrl.vec import RealEpisodeVecMonitor, RealEpisodeVecNormalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import (  # noqa: E402
    DEFAULT_APP, DEFAULT_PORT, MASTER_BUNDLE_ID, SAVE_ISOLATION_SUPPORTED,
    backup_saves, prepare_instance, preflight_ports, seed_save_dir,
)
from hkrl.cloneprep import prepare_clone_save  # noqa: E402

GAMMA = 0.995

# Fresh-run default for --target-kl. Both overnight 2-instance runs and
# hornet-1 destabilized late at approx_kl ~0.15-0.25; healthy early updates
# sat near 0.02. 0.03 (SB3 trips at 1.5x = 0.045) barely touches early
# learning while capping the tail. 0 disables the cap.
DEFAULT_TARGET_KL = 0.03

# Fraction of collected rows that are async-reset placeholders in a
# two-instance isolated run (measured on overnight-0723; see the
# RealEpisodeVecNormalize docstring in hkrl/vec.py). The gradient mask
# drops them from the loss, shrinking the effective batch by this much.
ASYNC_RESET_PLACEHOLDER_FRACTION = 0.19


def default_n_steps(instances: int, async_resets: bool = False) -> int:
    """Per-instance rollout length for --n-steps when not given explicitly.

    Divides so the total batch per update -- and with it the update's
    wall-clock time, which must stay inside the mod's 10s idle-disconnect
    ceiling -- holds at ~2048 whatever the fleet size. Floored at 128 so an
    absurd fleet still collects a usable sequence per instance.

    With async resets on, ~19% of the rows are placeholders the loss mask
    discards, so the rollout is inflated to keep REAL samples per update at
    ~2048 -- otherwise every N>=2 update learns from a smaller, noisier
    batch than N=1 at the same learning rate. The inflated batch makes the
    update ~23% longer (~8.3s at n_epochs=5); the keepalive pinger holds
    connections through it, but lower --n-epochs if updates crowd 10s.
    """
    total = 2048
    if async_resets:
        total = round(total / (1 - ASYNC_RESET_PLACEHOLDER_FRACTION))
    return max(128, total // instances)


def resolve_async_resets(flag, instances: int) -> bool:
    """Async resets are a multi-instance throughput feature: on by default
    at N>=2 since the Phase 2 gate passed, always off at N=1 (no sibling to
    freeze), and --no-async-resets is the escape hatch."""
    if instances < 2:
        return False
    return True if flag is None else bool(flag)


def resolve_timescale(value) -> float:
    """None (untyped, nothing recorded) means 1x; anything outside [1, 10]
    is refused rather than clamped -- the mod clamps as a last rail, but a
    typo on the CLI should fail loudly, not train at a surprise speed."""
    if value is None:
        return 1.0
    if not 1.0 <= value <= 10.0:
        sys.exit(f"--timescale must be between 1 and 10 (got {value})")
    return float(value)


def resolve_boss(flag: str | None, run_dir: Path | None) -> str:
    """The boss this session fights. Fresh runs take the flag (default:
    bosses.DEFAULT_BOSS). A resume takes the run's recorded boss -- the checkpoint's
    observation space is built from it, so it is not overridable: an
    explicit conflicting --boss is a hard error here, with a clear message
    instead of a shape mismatch deep inside model load. Configs from before
    the boss field read as DEFAULT_BOSS."""
    if run_dir is None:
        return flag or DEFAULT_BOSS
    configs = read_jsonl(run_dir / "config.jsonl")
    recorded = (configs[-1].get("boss") if configs else None) or DEFAULT_BOSS
    # A config naming a boss this registry lacks fails here, at the guard,
    # not deep in worker env construction.
    get_boss(recorded)
    if flag is not None and flag != recorded:
        raise ValueError(
            f"--boss {flag} conflicts with {run_dir}'s recorded boss "
            f"{recorded!r}; a checkpoint's observation space is built for "
            f"its boss, so a resume always keeps it. Start a new run to "
            f"train against {flag}.")
    return recorded


# On resume these inherit from the run's last config record unless the
# flag was typed: they are the settings still live on a resume (fleet
# shape, checkpoint cadence, update cap, window mode), where falling back
# to a CLI default silently reshapes the run -- the bug that dropped
# marmu-1 from 2 instances to 1. Each session re-records what it resolved,
# so headless sticks to whatever the LAST session used, not the launch.
# Session-specific flags (auto, measure_resets, root, app, run_id) and the
# checkpoint-baked model shape stay out.
RESUME_INHERITED = ("instances", "gen_every", "target_kl", "async_resets",
                    "async_reset_mode", "port", "headless", "timescale")

# Baked into the checkpoint zip: build_model ignores these on resume, so
# a typed flag would be a silent no-op -- refused instead, like a
# conflicting --boss.
RESUME_BAKED = ("n_steps", "batch_size", "n_epochs", "seed")


def apply_recorded_config(args, explicit: set, config: dict) -> None:
    """Layer a resume's settings: typed flag > recorded value > default.

    Mutates args in place. config is the run's last config.jsonl record
    ({} when the file is missing or empty); keys an old record lacks keep
    their CLI defaults. The recorded async_resets is the resolved boolean,
    which feeds resolve_async_resets exactly like an explicit flag would.
    """
    typed_baked = [k for k in RESUME_BAKED if k in explicit]
    if typed_baked:
        flags = ", ".join("--" + k.replace("_", "-") for k in typed_baked)
        raise ValueError(
            f"{flags}: PPO hyperparameters are baked into the checkpoint, "
            "so a resume keeps the recorded value and the flag would be "
            "silently ignored; remove it (only --target-kl can change a "
            "resumed run's update dynamics)")
    for key in RESUME_INHERITED:
        if key not in explicit and key in config:
            setattr(args, key, config[key])


def resolve_session_budget(timesteps: int, timesteps_typed: bool,
                           config: dict | None, current_timestep: int,
                           generations: list[dict]) -> tuple[int, int]:
    """This session's (learn budget, run target), in absolute timesteps.

    Fresh runs (config None): the budget is --timesteps and the target the
    same number. On resume an explicit --timesteps stays additive (collect
    N more); omitted, the session runs to the run's recorded
    target_timestep -- reconstructed additively from the last record for
    runs predating the key -- and a run already at its target refuses to
    start rather than silently extending by the flag's default.
    """
    if config is None:
        return timesteps, timesteps
    if timesteps_typed:
        return timesteps, current_timestep + timesteps
    target = config.get("target_timestep")
    if target is None and "timesteps" in config:
        # Pre-target_timestep record: rebuild the additive target its
        # session was launched with (rundata._target_timestep's walk).
        base = next((g["timestep"] for g in generations
                     if g["gen"] == config.get("resumed_from_gen")), 0)
        target = base + int(config["timesteps"])
    if target is None:
        print(f"hkrl: no recorded step target in this run's config; "
              f"collecting {timesteps:,} more steps (additive default)",
              file=sys.stderr, flush=True)
        return timesteps, current_timestep + timesteps
    target = int(target)
    if current_timestep >= target:
        raise ValueError(
            f"this run already reached its recorded target "
            f"({current_timestep:,} of {target:,} steps); pass "
            f"--timesteps N to extend it by N more steps")
    return target - current_timestep, target


def build_config_dict(args, async_resets, resume=None, started_at=None,
                      target_timestep=None, previous=None):
    """Build the config dict to be written to config.jsonl, recording the
    resolved async_resets value (not the raw tri-state flag) and the run's
    absolute step target. On resume, `previous` (the run's last record)
    supplies the checkpoint-baked values -- n_steps, batch_size, n_epochs,
    seed, gamma, ent_coef -- so the appended record states what the model
    actually trains with instead of this process's defaults."""
    if started_at is None:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    config = {
        **{k: str(v) if isinstance(v, Path) else v
           for k, v in vars(args).items()},
        # No "n_stack": the recurrent policy replaced frame stacking, so
        # there is no stack depth to record.
        "async_resets": async_resets,  # Override with resolved boolean
        "gamma": GAMMA, "ent_coef": 0.01,
        "target_timestep": target_timestep,
        "resumed_from_gen": resume[0] if resume else None,
        "started_at": started_at,
    }
    if resume and previous:
        for key in RESUME_BAKED + ("gamma", "ent_coef"):
            if key in previous:
                config[key] = previous[key]
    return config


def session_banner(timesteps: int, start_timestep: int = 0,
                   resumed_gen: int | None = None) -> str:
    """One line stating this session's budget in the dashboard's language:
    current timestep and the target it runs to (the budget is this
    session's resolved step count, so the target is start + budget)."""
    target = start_timestep + timesteps
    if resumed_gen is None:
        return (f"this session: collecting {timesteps:,} steps "
                f"(target timestep {target:,})")
    return (f"resumed from generation {resumed_gen} at timestep "
            f"{start_timestep:,}; collecting {timesteps:,} more "
            f"(target timestep {target:,})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=500_000,
                    help="env steps to collect this session (~54k/hour at "
                         "15 Hz). On resume the flag stays additive -- "
                         "collect N MORE steps -- but omitting it now "
                         "finishes to the run's recorded target instead of "
                         "adding the default")
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
                         "2048 // instances, inflated ~23%% when async "
                         "resets are on so the update still sees ~2048 REAL "
                         "samples after the placeholder mask). The default "
                         "divides so the total batch -- and with it the "
                         "update's wall-clock time -- stays roughly constant "
                         "as --instances grows: the games run on in real "
                         "time while the update computes, every Knight "
                         "standing in a live fight")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=5,
                    help="PPO epochs per update. Kept at 5 so the recurrent "
                         "256x256+LSTM update stays short (~6.7s on CPU): "
                         "the keepalive pinger keeps connections alive "
                         "through longer updates, but the Knights stand in "
                         "their live fights for the whole update. Raise "
                         "only if the net moves off CPU.")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="early-stop an update's remaining epochs once "
                         "approx_kl exceeds ~1.5x this (SB3 semantics). "
                         f"Fresh runs default to {DEFAULT_TARGET_KL} "
                         "(observed approx_kl ~0.15-0.25 without a cap "
                         "drives the late-run win-rate slide); 0 disables "
                         "the cap. Resumes inherit the run's recorded "
                         "value; typing the flag overrides the checkpoint, "
                         "unlike the other hyperparameters. The argparse "
                         "default stays None so inheritance can tell typed "
                         "from unset.")
    ap.add_argument("--boss", default=None, choices=sorted(BOSSES),
                    help=f"which boss to train against (default: {DEFAULT_BOSS}). "
                         "Sets the observation space, so checkpoints are "
                         "boss-specific: a resume always keeps the run's "
                         "recorded boss and refuses a conflicting flag.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--auto", action="store_true",
                    help="skip the interactive ready prompt (unattended/"
                         "dashboard launches); the boot macro drives the "
                         "game into the Hall of Gods")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="launch the game(s) with -batchmode -nographics: "
                         "no window, no occlusion App Nap risk; the mod "
                         "caps the frame loop at 60 fps. A resume keeps "
                         "the last session's value; --headless or "
                         "--no-headless overrides it.")
    ap.add_argument("--timescale", type=float, default=None,
                    help="run the game(s) at K x real time (1-10; the mod "
                         "multiplies Time.timeScale and scales its frame "
                         "cap, so a decision still spans ~66.7ms of game "
                         "time). A resume inherits the last session's "
                         "value; typing the flag overrides it "
                         "(--timescale 1 turns it off). The argparse "
                         "default stays None so inheritance can tell "
                         "typed from unset.")
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
    return ap


def parse_session_args(argv=None):
    """Parse argv and learn which flags the user actually typed.

    Returns (args, explicit): the parsed namespace plus the set of dests
    that appeared on the command line, learned from a second parse of the
    same argv with every default suppressed -- a dest surviving into that
    namespace can only have come from an actual flag. Resume inheritance
    (apply_recorded_config) needs exactly that typed-vs-default
    distinction.
    """
    args = build_parser().parse_args(argv)
    probe = build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    explicit = set(vars(probe.parse_args(argv)))
    return args, explicit


def build_env(ports, relaunch, run_dir, resume_vecnorm=None, **supervisor_kwargs):
    """SupervisedVecEnv -> RealEpisodeVecMonitor -> RealEpisodeVecNormalize.

    Returns (env, supervisor): the outermost wrapper for PPO, plus the
    supervisor itself so the checkpoint callback can read its recovery
    count.

    The monitor is RealEpisodeVecMonitor so isolated-mode async-reset
    throwaway episodes never reach the monitor CSV, the dashboard, or
    ep_rew_mean. It sits below the normalizer so its episode records carry
    raw rewards and true lengths. info_keywords puts won/boss_damage_frac in
    each episode's CSV row so the dashboard can color episodes by outcome;
    RealEpisodeVecMonitor skips keys a done step lacks (the supervisor's
    recovery frames carry only terminal_observation), leaving those columns
    blank. GenerationCallback still reads outcomes from the raw infos.

    No frame stacking: one observation is an instant, and the FSM one-hot
    does not encode how long the boss has been in a state -- but the recurrent
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
        supervisor, filename=str(Path(run_dir) / f"monitor_{session}"),
        info_keywords=("won", "boss_damage_frac"))
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
                n_steps=2048, batch_size=64, n_epochs=10, target_kl=None):
    """A RecurrentPPO for this env, fresh or loaded from a generation
    checkpoint.

    The masked subclass so async-reset placeholder transitions never reach
    the gradient (hkrl/masking.py); loading an old plain-RecurrentPPO
    checkpoint through it is fine, the weights are identical.

    On resume every hyperparameter comes from the checkpoint zip; the
    keyword arguments here shape fresh models only -- except target_kl,
    which when set overrides the checkpoint too, because the flag exists
    to change update dynamics on a run already in progress; a value of 0
    turns the cap off.
    """
    if resume_model is not None:
        model = MaskedRecurrentPPO.load(str(resume_model), env=env,
                                        device="cpu")
        if target_kl is not None:
            # 0 spells "cap off" in configs and argv; SB3 spells it None.
            model.target_kl = target_kl or None
        return model
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
        # Cap on update size (see DEFAULT_TARGET_KL): aborts an update's
        # remaining epochs once approx_kl exceeds ~1.5x this. 0 from the
        # flag/config means no cap, which SB3 spells as None.
        target_kl=target_kl or None,
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
    unwind a live boss through the truncation path -- the slowest,
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


def confirm_ready(auto: bool, boss_display: str) -> None:
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
    input(f"Bring the game(s) to the Hall of Gods near the {boss_display} "
          "statue, then press Enter. (A freshly booted game can also "
          "challenge itself in via the boot macro; expect a few reset "
          "retries.) ")


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


def build_prepares(ports):
    """Per-port relaunch callbacks that re-seed the clone save from master.

    A relaunch re-execs the same clone app, but the clone's SAVE can be
    the broken part: the wrong-save boot flake starts a new game when
    user1.dat reads empty/corrupt, and rebooting into that same save
    re-flakes until the recovery ceiling kills the run. Re-seeding on
    every relaunch makes the supervisor's existing relaunch path an
    actual cure. Returns None where clones don't exist (no save
    isolation); relaunch then behaves exactly as before.
    """
    if not SAVE_ISOLATION_SUPPORTED:
        return None

    def make(port):
        bundle_id = f"{MASTER_BUNDLE_ID}.hkrl{port}"
        return lambda: prepare_clone_save(seed_save_dir(bundle_id))

    return [make(p) for p in ports]


def prepare_session(argv=None) -> tuple[argparse.Namespace, Path, tuple | None, int, bool]:
    """Everything before any game process exists: parse and layer args,
    locate or create the run dir, resolve boss and step budget, and append
    this session's config record. Returns (args, run_dir, resume, budget,
    async_resets). Split from main() so the whole pre-flight -- including
    resume inheritance -- is testable without launching a game."""
    args, explicit = parse_session_args(argv)

    if args.resume is not None:
        run_dir = args.resume.expanduser()
        resume = latest_checkpoint(run_dir)  # (gen, weights, vecnorm)
        configs = read_jsonl(run_dir / "config.jsonl")
        previous = configs[-1] if configs else {}
        try:
            apply_recorded_config(args, explicit, previous)
        except ValueError as exc:
            sys.exit(str(exc))
    else:
        resume = None
        previous = None
        run_dir = args.root / "runs" / (args.run_id
                                        or time.strftime("%Y%m%d_%H%M%S"))

    if args.instances < 1:
        sys.exit("--instances must be at least 1")
    # Warn only for a TYPED --async-resets at N=1: an inherited True is
    # silently forced off by resolve_async_resets below, as designed.
    if (args.async_resets is True and "async_resets" in explicit
            and args.instances < 2):
        print("hkrl: --async-resets is a no-op at --instances 1 (no sibling "
              "to freeze); running synchronously", file=sys.stderr, flush=True)
    # Resolved before the config dump below so config.jsonl records the
    # values actually used, not None. async_resets first: the n_steps
    # default inflates to cover the placeholder rows it masks out.
    async_resets = resolve_async_resets(args.async_resets, args.instances)
    if args.n_steps is None:
        args.n_steps = default_n_steps(args.instances, async_resets)
    args.timescale = resolve_timescale(args.timescale)

    # Fresh runs get the update cap by default; the resolved value lands in
    # the config dump below so resumes inherit it. Resume path untouched:
    # a pre-default run recorded null and stays uncapped (resume is a
    # continuation), until --target-kl is typed once.
    if resume is None and args.target_kl is None:
        args.target_kl = DEFAULT_TARGET_KL

    try:
        args.boss = resolve_boss(args.boss,
                                 run_dir if args.resume is not None else None)
    except ValueError as exc:
        sys.exit(str(exc))

    generations = read_jsonl(run_dir / "generations.jsonl")
    current = (next((g["timestep"] for g in generations
                     if g["gen"] == resume[0]), 0) if resume else 0)
    try:
        budget, target = resolve_session_budget(
            args.timesteps, "timesteps" in explicit, previous, current,
            generations)
    except ValueError as exc:
        sys.exit(str(exc))
    # The record must state what this session actually collects: on an
    # untyped finish-to-target resume args.timesteps still holds the
    # argparse default, which would misstate the session. Fresh runs and
    # explicit resumes are no-ops (budget == the flag there). An inherited
    # target_kl is also re-applied to the loaded model via build_model --
    # idempotent, the checkpoint trained under that same recorded value.
    args.timesteps = budget

    # The dir is created only after every validation above passed, so bad
    # args never leave a junk run dir behind (same guarantee main() gave
    # by validating before its mkdir).
    if resume is None:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            sys.exit(f"{run_dir} already exists. Restarting into an existing "
                     f"run is never implicit: pass --resume {run_dir} to "
                     f"continue it, or a different --run-id to start fresh.")

    # One JSON object per session, appended, so a resumed run's full history
    # stays inspectable next to its checkpoints.
    with (run_dir / "config.jsonl").open("a") as f:
        config = build_config_dict(args, async_resets, resume=resume,
                                   target_timestep=target, previous=previous)
        f.write(json.dumps(config) + "\n")
    return args, run_dir, resume, budget, async_resets


def main() -> None:
    args, run_dir, resume, budget, async_resets = prepare_session()

    ports = [args.port + i for i in range(args.instances)]
    # Fail fast on squatted ports BEFORE the save backup and the clone
    # work: the PortInUse that used to fire at spawn time arrived after
    # both, buried under a traceback (observed 2026-08-22 -- three launches
    # died on leftover games squatting 9021-9023). Dashboard launches are
    # normally refused even earlier, by launcher._preflight.
    verdicts = preflight_ports(ports)
    if verdicts:
        for v in verdicts:
            print(f"!!! {v}", file=sys.stderr, flush=True)
        print(f"!!! nothing was launched. This session is already recorded "
              f"in {run_dir}; free the port(s), then resume the run (or "
              f"remove that dir to start it fresh).",
              file=sys.stderr, flush=True)
        sys.exit(1)

    # Unconditional: every clone is seeded FROM the master save (N=1 on
    # save-isolation platforms, the master itself on others), so a quietly
    # corrupt master would propagate. A run must never be the event that
    # loses someone's save.
    backup = backup_saves(args.root)
    if backup is not None:
        print(f"master save backed up to {backup}", flush=True)
    if resume is None:
        print(session_banner(budget), flush=True)

    # Instances must not share the game's save directory: they all autosave
    # the same slot throughout a run (observed corrupting the master save
    # live, 2026-07-20 -- see seed_save_dir). Each slot gets its own app
    # clone with a per-port bundle id (own save dir, own ModLog), refreshed
    # from the master app and save at every start -- see build_apps.
    apps = build_apps(ports, args.app, args.root / "instances")
    game = GameFleet(ports, app=args.app, apps=apps,
                     prepares=build_prepares(ports),
                     headless=args.headless, timescale=args.timescale)
    env = None
    exit_code = 0
    try:
        game.start()
        print(f"game(s) up on port(s) {', '.join(map(str, game.ports))}",
              flush=True)
        if args.headless:
            print("headless: no game window will appear; the mod caps the "
                  "frame loop at 60 fps. Keep sleep suppressed for the run "
                  "(caffeinate -dims on macOS).", flush=True)
        elif sys.platform == "win32":
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
        if args.timescale > 1.0:
            print(f"timescale: game running at {args.timescale}x real time "
                  "(watch ModLog's per-episode speed= ratio; below "
                  f"{args.timescale} means the machine can't keep up)",
                  flush=True)
        confirm_ready(args.auto, get_boss(args.boss).display_name)

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
            boss=args.boss,
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
                            batch_size=args.batch_size, n_epochs=args.n_epochs,
                            target_kl=args.target_kl)
        if resume:
            print(f"{run_dir}: " + session_banner(
                budget, start_timestep=model.num_timesteps,
                resumed_gen=resume[0]), flush=True)
        callback = GenerationCallback(run_dir, vecnorm=env,
                                      every_steps=args.gen_every,
                                      supervisor=supervisor)
        try:
            model.learn(total_timesteps=budget,
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
