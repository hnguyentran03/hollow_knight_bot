"""Launch N isolated Hollow Knight instances for parallel training.

Each instance gets a private HOME (so Unity's save directory and ModLog.txt
are per-instance) and a private HKRL_PORT. The game binary is executed
directly rather than through Steam, which refuses to launch a second copy.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.instances import port_for, provision  # noqa: E402

DEFAULT_APP = Path(
    "~/Library/Application Support/Steam/steamapps/common/Hollow Knight/"
    "hollow_knight.app/Contents/MacOS/Hollow Knight"
).expanduser()

DEFAULT_SEED = Path(
    "~/Library/Application Support/unity.Team Cherry.Hollow Knight"
).expanduser()


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
    matters because the waits are sequential and instances 1..N-1 run with
    stdout/stderr on DEVNULL, so a bad --app path or a mod that fails to load
    is otherwise silent for timeout * instances before the first error.

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


def launch(n: int, home: Path, port: int, app: Path, visible: bool) -> subprocess.Popen:
    env = dict(os.environ, HOME=str(home), HKRL_PORT=str(port))
    return subprocess.Popen(
        [str(app)],
        env=env,
        stdout=subprocess.DEVNULL if not visible else None,
        stderr=subprocess.DEVNULL if not visible else None,
    )


def shutdown(procs: list, grace: float = 15.0) -> None:
    """Terminate every process and block until each has actually exited.

    A game process runs its own Unity/Mono save-on-exit routine after
    SIGTERM, which can take several seconds; 15s is generous enough to let
    that finish under normal conditions. Any process still alive once the
    shared deadline passes is SIGKILLed and reaped, so a hung or
    signal-ignoring process can never outlive this call and be left holding
    the instance's HOME or bridge port for a subsequent launch to collide
    with.
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
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser())
    ap.add_argument("--seed-from", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--app", type=Path, default=DEFAULT_APP)
    args = ap.parse_args()

    procs = []
    try:
        for n in range(args.instances):
            home = provision(n, root=args.root, seed_from=args.seed_from)
            port = port_for(n)
            # All instances open a game window; `visible` only controls
            # whether that instance's stdout/stderr reach this terminal.
            # Instance 0 keeps them so the human sees mod output from one
            # instance; the rest are silenced to keep the log readable.
            procs.append(launch(n, home, port, args.app, visible=(n == 0)))
            print(f"instance {n}: home={home} port={port}", flush=True)

        for n in range(args.instances):
            wait_for_port(port_for(n), proc=procs[n])
            print(f"instance {n}: bridge ready on {port_for(n)}", flush=True)

        print(f"{args.instances} instances ready. Ctrl-C to stop.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(procs)


if __name__ == "__main__":
    main()
