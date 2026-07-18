#!/usr/bin/env python3
"""Random agent: proves mod + protocol + env end-to-end. Run the game first.

Connects to the running HKRLBot mod, plays `--episodes` episodes with a
uniformly random policy over `hkrl.env.ACTIONS`, and prints one summary line
per episode.

This is also the instrument used to verify -- live, for the first time --
the two mod behaviors nothing else in the pipeline exercises: the
death-retry auto-confirm macro and the win-path statue walk (see
mod/EpisodeManager.cs's ResetMacro). Both are pulse-timing guesses against
UI nobody has watched the mod drive yet, so this script is written to make
three things visually obvious without the human having to guess:

  1. Episodes are actually completing (steps/reward/won printed each time).
  2. Resets are actually happening (the `attempt` number must strictly
     increase episode to episode; a repeat is flagged inline).
  3. A stall is distinguishable from slow progress: reset() and step() are
     both wrapped with a watchdog that prints "... still waiting" heartbeats
     if the mod goes quiet for longer than is normal for that call, well
     before the protocol's 30s socket timeout could fire. A real wedge
     shows up as a growing run of heartbeats; a genuine crash/disconnect is
     reported by name instead of a raw traceback.
"""
import argparse
import socket
import sys
import time
import threading

sys.path.insert(0, ".")

from hkrl.env import HKEnv, OBS_KEYS
from hkrl.protocol import ConnectionClosed

# How long to wait before printing the first "still waiting" heartbeat for a
# blocked call, and how often to repeat it after that.
#
# step() is one 15 Hz decision cycle (~67ms of held input under normal
# lockstep), so a couple of seconds of silence is already suspicious.
# reset() can legitimately run the death-retry confirm pulse (~2s cycle) or
# the win-path statue walk + challenge-menu macro (~3s cycle, plus however
# long the walk to the statue takes), so it gets a longer grace period.
# A reset() that truncates a still-live fight (HKEnv's max_steps) waits for
# that fight to actually end before the mod will accept the reset as fresh
# -- this can legitimately take tens of seconds, bounded only by the mod's
# own ResetMacroBudgetTicks backstop (mod/EpisodeManager.cs; 22.5s as of
# this writing), which is itself kept below hkrl.protocol.Connection's 30s
# socket timeout so the mod always gives up and logs first. A real stall
# still prints several heartbeats before that timeout could fire.
STEP_WARN_AFTER = 2.0
STEP_WARN_EVERY = 2.0
RESET_WARN_AFTER = 5.0
RESET_WARN_EVERY = 5.0

BHP_INDEX = OBS_KEYS.index("bhp")


class Wedge(Exception):
    """The mod stopped responding entirely (socket timeout) or dropped the
    connection mid-episode. Carries human-readable context (which episode,
    how long it had been waiting) so the driver's output names the failure
    instead of surfacing a raw traceback.

    `summaries` (set by run(), not by run_episode()) holds whichever
    earlier episodes in the same run() call already completed
    successfully, so a caller that catches Wedge can still report an
    accurate final tally instead of silently losing episodes that in fact
    finished fine before the wedge.
    """

    def __init__(self, message, summaries=None):
        super().__init__(message)
        self.summaries = summaries if summaries is not None else []


def _with_heartbeat(fn, label, warn_after, warn_every, out=None):
    """Run fn() on the calling thread; a watchdog thread prints a heartbeat
    to `out` (default sys.stderr, resolved at call time) if it takes longer
    than `warn_after` seconds, then every `warn_every` seconds after that,
    until fn() returns or raises. This is what makes a stall visually
    distinct from slow-but-fine progress: normal calls print nothing at
    all; a slow one prints a couple of heartbeats; a wedged one prints them
    indefinitely until the socket times out.
    """
    if out is None:
        out = sys.stderr
    done = threading.Event()
    start = time.monotonic()

    def watchdog():
        wait = warn_after
        while not done.wait(wait):
            elapsed = time.monotonic() - start
            print(f"  ... still waiting on {label} "
                  f"({elapsed:.1f}s elapsed, no response from mod)",
                  file=out, flush=True)
            wait = warn_every

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()
        t.join()


def run_episode(env, ep_num,
                 step_warn_after=STEP_WARN_AFTER, step_warn_every=STEP_WARN_EVERY,
                 reset_warn_after=RESET_WARN_AFTER, reset_warn_every=RESET_WARN_EVERY,
                 out=None):
    """Play one episode with a uniformly random policy. Prints a one-line
    summary and returns it as a dict. Raises Wedge if the mod stops
    responding or drops the connection mid-episode.
    """
    if out is None:
        out = sys.stdout
    start = time.monotonic()
    try:
        obs, info = _with_heartbeat(
            env.reset, "reset() [death-retry / win-path macro]",
            reset_warn_after, reset_warn_every)
    except socket.timeout as e:
        raise Wedge(
            f"episode {ep_num}: mod stopped responding to reset() after "
            f"{time.monotonic() - start:.1f}s (socket timed out) -- the "
            f"death-retry or win-path macro likely hung") from e
    except ConnectionClosed as e:
        raise Wedge(
            f"episode {ep_num}: mod closed the connection during reset() "
            f"after {time.monotonic() - start:.1f}s -- check the in-game "
            f"console/log for an exception") from e

    total, steps = 0.0, 0
    min_bhp_frac = None
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        try:
            obs, r, terminated, truncated, info = _with_heartbeat(
                lambda: env.step(action), "step()",
                step_warn_after, step_warn_every)
        except socket.timeout as e:
            raise Wedge(
                f"episode {ep_num}: mod stopped responding to step() after "
                f"{time.monotonic() - start:.1f}s total, {steps} steps in "
                f"(socket timed out)") from e
        except ConnectionClosed as e:
            raise Wedge(
                f"episode {ep_num}: mod closed the connection mid-episode "
                f"after {time.monotonic() - start:.1f}s total, {steps} "
                f"steps in -- check the in-game console/log for an "
                f"exception") from e
        total += r
        steps += 1
        bhp_frac = float(obs[BHP_INDEX])
        if min_bhp_frac is None or bhp_frac < min_bhp_frac:
            min_bhp_frac = bhp_frac

    elapsed = time.monotonic() - start
    won = bool(info.get("won", False))
    result = "WIN" if won else ("TIMEOUT" if truncated else "loss")
    bhp_str = "?" if min_bhp_frac is None else f"{min_bhp_frac * 100:5.1f}%"
    summary = dict(episode=ep_num, attempt=info.get("attempt"), steps=steps,
                    reward=total, won=won, truncated=truncated,
                    terminated=terminated, min_bhp_frac=min_bhp_frac,
                    wall_s=elapsed, scene=info.get("scene"))
    print(f"episode {ep_num:3d} (attempt {info.get('attempt')}): "
          f"result={result:7s} steps={steps:4d} reward={total:8.2f} "
          f"boss_hp_min={bhp_str} wall={elapsed:5.1f}s scene={info.get('scene')}",
          file=out, flush=True)
    return summary


def run(env, episodes, out=None, **heartbeat_kwargs):
    """Run `episodes` episodes end to end, printing a summary per episode
    plus an inline warning if `attempt` fails to strictly increase (a sign
    that a reset "completed" from the client's point of view without the
    mod actually restarting the fight). Returns the list of per-episode
    summary dicts. Propagates Wedge if the mod stalls or disconnects.
    """
    if out is None:
        out = sys.stdout
    summaries = []
    prev_attempt = None
    try:
        for ep in range(1, episodes + 1):
            summary = run_episode(env, ep, out=out, **heartbeat_kwargs)
            attempt = summary["attempt"]
            if prev_attempt is not None and attempt is not None and attempt <= prev_attempt:
                print(f"  WARNING: attempt did not advance (still {attempt}, "
                      f"previously {prev_attempt}) -- the reset may not have "
                      f"actually completed even though the episode finished",
                      file=out, flush=True)
            prev_attempt = attempt
            summaries.append(summary)
    except Wedge as e:
        # Preserve whatever completed successfully before the wedge, so a
        # caller reports an accurate tally instead of losing episodes that
        # in fact finished fine (see Wedge's docstring).
        e.summaries = summaries
        raise
    return summaries


def _print_final_summary(summaries, requested, out=None):
    if out is None:
        out = sys.stdout
    n = len(summaries)
    wins = sum(1 for s in summaries if s["won"])
    attempts = [s["attempt"] for s in summaries]
    print(f"\n{n}/{requested} episodes completed, {wins} won, "
          f"attempts seen: {attempts}", file=out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9020)
    ap.add_argument("--seed", type=int, default=None,
                     help="seed the random action sampler for a reproducible run")
    args = ap.parse_args()

    # Final-review fix (F7): this is the most likely first-run failure --
    # game not running, mod not installed/loaded, or a wrong --port -- and
    # without this it surfaces as a raw ConnectionRefusedError traceback with
    # nothing pointing the human at the actual cause.
    try:
        env = HKEnv(host=args.host, port=args.port)
    except ConnectionRefusedError:
        print(f"\n!!! Could not connect to the HKRLBot mod at "
              f"{args.host}:{args.port}.\n"
              f"!!! Likely cause: the game isn't running, the mod isn't "
              f"installed/loaded, or --port is wrong.\n"
              f"!!! Start Hollow Knight with the mod installed "
              f"(see mod/build.sh) and try again.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        env.action_space.seed(args.seed)

    summaries = []
    try:
        summaries = run(env, args.episodes)
    except Wedge as e:
        summaries = e.summaries
        print(f"\n!!! WEDGE: {e}", file=sys.stderr)
        print("!!! The driver cannot continue with a dead connection -- "
              "restart the game/mod and re-run.", file=sys.stderr)
        _print_final_summary(summaries, args.episodes)
        env.close()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        _print_final_summary(summaries, args.episodes)
        env.close()
        sys.exit(130)

    _print_final_summary(summaries, args.episodes)
    env.close()


if __name__ == "__main__":
    main()
