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


def wait_for_port(port: int, timeout: float = 120.0, host: str = "127.0.0.1") -> None:
    """Block until the port accepts a connection.

    The bridge only starts listening once the mod has loaded, which is well
    after the process itself exists -- so process liveness is not a usable
    readiness signal.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
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
            # Instance 0 stays visible as the human's window into the agent.
            procs.append(launch(n, home, port, args.app, visible=(n == 0)))
            print(f"instance {n}: home={home} port={port}", flush=True)

        for n in range(args.instances):
            wait_for_port(port_for(n))
            print(f"instance {n}: bridge ready on {port_for(n)}", flush=True)

        print(f"{args.instances} instances ready. Ctrl-C to stop.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
