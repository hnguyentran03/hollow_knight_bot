"""Launch a Hollow Knight instance with the RL bridge and wait for it.

The game binary is executed directly rather than through Steam so the launch
is a plain child process this script (and the supervisor) can signal and reap.
HKRL_PORT tells the mod which port to listen on.

Also the library the supervisor and the training script import: `launch`,
`shutdown` and `wait_for_port` are the three calls a `relaunch(slot)` needs.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    DEFAULT_APP = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight"
        r"\hollow_knight.exe"
    )
else:
    DEFAULT_APP = Path(
        "~/Library/Application Support/Steam/steamapps/common/Hollow Knight/"
        "hollow_knight.app/Contents/MacOS/Hollow Knight"
    ).expanduser()

DEFAULT_PORT = 9020


def wait_for_port(
    port: int,
    timeout: float = 120.0,
    host: str = "127.0.0.1",
    proc: subprocess.Popen | None = None,
) -> None:
    """Block until the port accepts a connection.

    The bridge only starts listening once the mod has loaded, which is well
    after the process itself exists -- so process liveness is not a usable
    *readiness* signal. It is a usable *failure* signal: a process that has
    already exited will never listen, so with `proc` supplied this raises at
    once naming the exit code instead of burning the full timeout. That
    matters because a game launched with stdout/stderr on DEVNULL reports a
    bad --app path or a mod that fails to load in no other way.

    `proc` is optional so callers holding only a port keep working.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            # Checked after the connect attempt, not before: a process that
            # exited *after* handing off a working bridge is not the case
            # being caught here, and connecting first keeps that ordering
            # from mattering.
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"process for port {port} exited with code "
                    f"{proc.returncode} before its bridge started listening"
                )
            time.sleep(0.5)
    raise TimeoutError(f"port {port} never accepted a connection within {timeout}s")


def launch(port: int, app: Path, visible: bool) -> subprocess.Popen:
    # SteamAppId/SteamGameId are the launch context Steam exports to its
    # children; the Steam build's DRM check looks for them in the
    # environment. Executed directly without them, the game asks Steam to
    # relaunch it (steam://run/367520 in Steam's console log) and quits
    # ~15s after boot -- at the title menu, before any bridge traffic.
    # steam_appid.txt beside the binary or in the cwd does NOT satisfy the
    # check on macOS; only the env vars do (verified live, 2026-07-19).
    # Unverified on Windows, where steam_appid.txt reportedly also works;
    # the env vars are kept as the single cross-platform mechanism.
    env = dict(os.environ, HKRL_PORT=str(port),
               SteamAppId="367520", SteamGameId="367520")
    # Detach the game from the terminal's Ctrl-C: Ctrl-C at the trainer must
    # interrupt the trainer alone, not kill the game out from under the
    # supervisor and the final checkpoint save. shutdown() is the one
    # intended kill path. On POSIX that means a new session (out of the
    # terminal's foreground process group); on Windows, a new process group
    # (out of the console's CTRL_C_EVENT broadcast).
    if sys.platform == "win32":
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        detach = {"start_new_session": True}
    return subprocess.Popen(
        [str(app)],
        env=env,
        stdout=subprocess.DEVNULL if not visible else None,
        stderr=subprocess.DEVNULL if not visible else None,
        **detach,
    )


def shutdown(procs: list, grace: float = 15.0) -> None:
    """Terminate every process and block until each has actually exited.

    A game process runs its own Unity/Mono save-on-exit routine after
    SIGTERM, which can take several seconds; 15s is generous enough to let
    that finish under normal conditions. Any process still alive once the
    shared deadline passes is SIGKILLed and reaped, so a hung or
    signal-ignoring process can never outlive this call and be left holding
    the bridge port for a subsequent launch to collide with.

    On Windows, terminate() IS TerminateProcess -- an immediate hard kill
    with no save-on-exit window. That is acceptable here: the training save
    is parked at the Hall of Gods bench and the mod never writes game saves,
    so there is nothing in memory worth flushing. The escalation path is
    then a no-op (kill() == terminate()), but the wait-and-reap contract
    holds identically.
    """
    for p in procs:
        p.terminate()
    deadline = time.monotonic() + grace
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=DEFAULT_APP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    procs = []
    try:
        procs.append(launch(args.port, args.app, visible=True))
        print(f"launched: port={args.port}", flush=True)
        wait_for_port(args.port, proc=procs[0])
        print(f"bridge ready on {args.port}. Ctrl-C to stop.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(procs)


if __name__ == "__main__":
    main()
