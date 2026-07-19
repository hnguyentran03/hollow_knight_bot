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
import time
from pathlib import Path

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
    env = dict(os.environ, HKRL_PORT=str(port))
    return subprocess.Popen(
        [str(app)],
        env=env,
        stdout=subprocess.DEVNULL if not visible else None,
        stderr=subprocess.DEVNULL if not visible else None,
        # A new session detaches the game from the terminal's process group:
        # Ctrl-C at the trainer must interrupt the trainer alone, not kill
        # the game out from under the supervisor and the final checkpoint
        # save. shutdown() is the one intended kill path.
        start_new_session=True,
    )


def shutdown(procs: list, grace: float = 15.0) -> None:
    """Terminate every process and block until each has actually exited.

    A game process runs its own Unity/Mono save-on-exit routine after
    SIGTERM, which can take several seconds; 15s is generous enough to let
    that finish under normal conditions. Any process still alive once the
    shared deadline passes is SIGKILLed and reaped, so a hung or
    signal-ignoring process can never outlive this call and be left holding
    the bridge port for a subsequent launch to collide with.
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
