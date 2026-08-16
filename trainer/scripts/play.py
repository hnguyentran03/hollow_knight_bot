#!/usr/bin/env python3
"""Play any exported bot in the live game, one episode per in-game F9.

Run the game (mod on port 9020) and this daemon once, in either order.
Pick a bot in Options -> Mods -> HKRLBot -- the selection rides the mod's
F9 play event, so this script takes no bot argument -- then press F9 at
the Hall of Gods. The daemon loads that export (cached after first use),
switches the env to the export's boss, runs exactly ONE episode, and goes
back to idle; control returns to the human when the fight ends.

Deliberately not a VecEnv pipeline: DummyVecEnv autoresets, which would
send a new reset -- and start a whole new fight -- the moment an episode
ends. This manual loop sends reset only on a play event, never after.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.env import HKEnv  # noqa: E402
from hkrl.exports import (EXPORT_MANIFEST, EXPORTS_DIR, MODEL_NAME,  # noqa: E402
                          VECNORM_NAME)
from hkrl.protocol import ConnectionClosed, PROTOCOL_VERSION  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import DEFAULT_PORT, backup_saves  # noqa: E402


class ExportError(Exception):
    """A per-play failure: reported on stderr, the daemon stays up."""


def _load_model(path):
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO.load(str(path), device="cpu")


def _load_vecnorm(path):
    # Standalone unpickle, no VecEnv wrapping: only the frozen obs_rms /
    # normalize_obs() are used, exactly like replay's training=False.
    with open(path, "rb") as f:
        return pickle.load(f)


def load_export(root, name, cache, load_model=_load_model,
                load_vecnorm=_load_vecnorm):
    """{"manifest","model","vecnorm"} for an export, cached per name so
    only the first F9 after a switch pays the model load."""
    if name in cache:
        return cache[name]
    exports_dir = Path(root).expanduser() / EXPORTS_DIR
    d = exports_dir / name
    manifest_path = d / EXPORT_MANIFEST
    if not d.is_dir() or not manifest_path.exists():
        raise ExportError(
            f"no export named {name!r} under {exports_dir} -- does the "
            f"mod's HKRL_ROOT match this daemon's --root?")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"unreadable manifest for {name!r}: {exc}")
    model_path, vecnorm_path = d / MODEL_NAME, d / VECNORM_NAME
    if not model_path.exists():
        raise ExportError(f"export {name!r} is missing {MODEL_NAME} -- a "
                          f"half-deleted export? re-export it")
    if not vecnorm_path.exists():
        raise ExportError(f"export {name!r} is missing {VECNORM_NAME} -- a "
                          f"half-deleted export? re-export it")
    entry = {"manifest": manifest,
             "model": load_model(model_path),
             "vecnorm": load_vecnorm(vecnorm_path)}
    cache[name] = entry
    return entry


def play_episode(env, model, vecnorm, name, deterministic=True,
                 out=None):
    """Exactly one episode; returns a replay-style summary dict.

    reset() is sent once, up front (the mod's macro walks to the statue
    and starts the fight); NOTHING is sent after the episode ends, so the
    mod goes idle and the human gets control back. LSTM handling mirrors
    replay.py: None state + episode_start=True initializes, and feeding
    done back in would re-zero it -- moot here, the loop ends at done.
    """
    if out is None:
        out = sys.stdout
    obs, _ = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    total, steps = 0.0, 0
    done = truncated = False
    info = {}
    while not (done or truncated):
        action, lstm_states = model.predict(
            vecnorm.normalize_obs(obs[None]), state=lstm_states,
            episode_start=episode_start, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(int(action[0]))
        episode_start = np.array([done or truncated])
        total += float(reward)
        steps += 1
    won = bool(info.get("won", False))
    result = "WIN" if won else ("TIMEOUT" if truncated else "loss")
    damage = float(info.get("boss_damage_frac", 0.0))
    print(f"[{name}] result={result:7s} steps={steps:4d} "
          f"reward={total:8.2f} boss_dmg={damage * 100:5.1f}%",
          file=out, flush=True)
    return dict(result=result, won=won, steps=steps, reward=total,
                boss_damage_frac=damage)


def handle_event(msg, *, root, env, cache, deterministic=True,
                 out=None, load_model=_load_model,
                 load_vecnorm=_load_vecnorm):
    """One idle-loop message. Returns the episode summary for a completed
    play, None for everything else. Per-play failures print one line and
    leave the daemon idle -- never a crash, never a reset."""
    if out is None:
        out = sys.stdout
    if msg.get("type") != "event" or msg.get("name") != "play":
        return None
    name = msg.get("bot")
    if not name:
        print("play event without a bot name -- is a bot selected in the "
              "in-game mod menu?", file=sys.stderr, flush=True)
        return None
    try:
        entry = load_export(root, name, cache, load_model=load_model,
                            load_vecnorm=load_vecnorm)
        boss = entry["manifest"].get("boss")
        if not boss:
            raise ExportError(
                f"manifest for {name!r} has no boss field -- re-export it "
                f"with the current trainer")
        env.set_boss(boss)   # ValueError on an id this trainer doesn't know
    except (ExportError, ValueError) as exc:
        print(f"cannot play {name!r}: {exc}", file=sys.stderr, flush=True)
        return None
    m = entry["manifest"]
    stats = m.get("stats") or {}
    print(f"playing {name} (gen {m.get('gen')} of {m.get('run_id')}, "
          f"{m.get('boss_display') or boss}, "
          f"win_rate {stats.get('win_rate', 0):.0%})", file=out, flush=True)
    # A stray F9 buffered mid-episode must not surface inside env.step's
    # recv() (its message shape doesn't match a step reply); filter events
    # for the duration of the fight, same as during training.
    env.conn.accept_events = False
    try:
        return play_episode(env, entry["model"], entry["vecnorm"], name=name,
                            deterministic=deterministic, out=out)
    finally:
        env.conn.accept_events = True


def _reconnect(env, out=None):
    """Re-dial the same Connection until the game is back."""
    if out is None:
        out = sys.stdout
    while True:
        env.conn.close()
        try:
            env.conn.connect()
        except (OSError, ConnectionClosed):
            time.sleep(2.0)
            continue
        version = (env.conn.hello or {}).get("version")
        if version != PROTOCOL_VERSION:
            sys.exit(f"mod speaks protocol v{version}, this daemon needs "
                     f"v{PROTOCOL_VERSION} -- rebuild the mod "
                     f"(mod/build.sh) and restart the game")
        print("reconnected", file=out, flush=True)
        return


def idle_loop(env, root, cache, deterministic=True, out=None,
              reconnect=None, load_model=_load_model,
              load_vecnorm=_load_vecnorm):
    """Dispatch bridge traffic until interrupted. Timeouts just loop (idle
    is quiet by design -- the keepalive pinger's pongs are filtered inside
    recv); a closed connection re-enters the wait-for-game loop; each play
    event runs exactly one episode."""
    if out is None:
        out = sys.stdout
    if reconnect is None:
        reconnect = _reconnect
    env.conn.accept_events = True
    while True:
        try:
            msg = env.conn.recv()
        except TimeoutError:      # socket.timeout is an alias since 3.10
            continue
        except ConnectionClosed:
            print("game connection lost; waiting for it to come back…",
                  file=out, flush=True)
            reconnect(env, out)
            env.conn.accept_events = True
            continue
        try:
            handle_event(msg, root=root, env=env, cache=cache,
                         deterministic=deterministic, out=out,
                         load_model=load_model, load_vecnorm=load_vecnorm)
        except ConnectionClosed:
            # The game closed mid-fight (env.step raised it): same
            # recovery as a closed recv() above.
            print("game connection lost; waiting for it to come back…",
                  file=out, flush=True)
            reconnect(env, out)
            env.conn.accept_events = True
        except Exception as exc:   # never BaseException: Ctrl-C must work
            print(f"unexpected error handling a play event: {exc}",
                  file=sys.stderr, flush=True)


def connect_env(host, port, out=None):
    """Construct the one HKEnv (it connects in __init__) in a retry loop,
    so the daemon can start before the game. A protocol-version mismatch
    (RuntimeError) propagates -- that needs a rebuild, not patience."""
    if out is None:
        out = sys.stdout
    printed = False
    while True:
        try:
            return HKEnv(host=host, port=port)
        except (OSError, ConnectionClosed):
            if not printed:
                print(f"waiting for game on port {port}…", file=out,
                      flush=True)
                printed = True
            time.sleep(2.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser(),
                    help="hkrl root: exports live under <root>/exports, "
                         "and the master save is backed up under it")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of argmax actions")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser()
    backup = backup_saves(root)
    if backup is not None:
        print(f"master save backed up to {backup}", flush=True)
    env = connect_env(args.host, args.port)
    print("connected -- select a bot in Options > Mods > HKRLBot and "
          "press F9 in the Hall of Gods (Ctrl-C quits)", flush=True)
    try:
        idle_loop(env, root, cache={},
                  deterministic=not args.stochastic)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
