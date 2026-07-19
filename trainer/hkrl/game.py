"""Owns the training run's game process: launch, relaunch, shut down.

train.py runs the game as its own child rather than beside a separately
started launcher: the supervisor's relaunch(slot) callback has to terminate
and reap the game currently holding the port, and a process can only reap
its own children. A relaunch driven from outside the launching process would
orphan a game that launcher's shutdown() never sees, leaving a zombie bound
to the port.
"""
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

# scripts/ is not a package; reached via a path insert, same convention as
# hkrl/supervisor.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_instances import (  # noqa: E402
    DEFAULT_APP, DEFAULT_PORT, launch, shutdown, wait_for_port,
)


class PortInUse(RuntimeError):
    """The game's port already accepts connections before any launch."""


class GameProcess:
    """The single game process behind one training run.

    The game's stdout/stderr go to DEVNULL (visible=False): SB3 writes its
    progress tables to the trainer's stdout, and the game's real diagnostic
    channel is ModLog.txt in the Unity save directory.
    """

    def __init__(self, port: int = DEFAULT_PORT, app: Path = DEFAULT_APP,
                 launch: Callable = launch, shutdown: Callable = shutdown,
                 wait_for_port: Callable = wait_for_port,
                 launch_timeout: float = 120.0):
        self.port = port
        self.app = Path(app)
        self.launch_timeout = launch_timeout
        self._launch = launch
        self._shutdown = shutdown
        self._wait_for_port = wait_for_port
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        """Launch the game and wait for its bridge to accept.

        Fails fast on a squatter: this process can only reap its own
        children, so a port held by a leftover game from an earlier run
        would make every relaunch collide (the mod's TcpListener.Start()
        throws on an in-use port) while the readiness probe happily greets
        the zombie.
        """
        if _accepting(self.port):
            raise PortInUse(
                f"port {self.port} is already accepting connections before "
                f"any game was launched -- an unmanaged process (a leftover "
                f"from an earlier run?) holds it. Find it with "
                f"`lsof -nP -iTCP:{self.port} -sTCP:LISTEN`, kill it, rerun."
            )
        self._proc = self._launch(self.port, self.app, False)
        self._wait_for_port(self.port, timeout=self.launch_timeout,
                            proc=self._proc)

    def relaunch(self, slot: int = 0) -> None:
        """SupervisedVecEnv's relaunch callback (slot is always 0 at N=1).

        Terminates and reaps the current holder BEFORE launching -- the
        supervisor's documented contract: a wedged or App-Nap-suspended game
        still holds its port, so launching first would make the replacement's
        listener throw while the probe greets the zombie. shutdown()'s
        SIGKILL escalation is what reaps a suspended process (SIGTERM stays
        pending on a stopped process). Does not wait for the new bridge; the
        supervisor runs its own wait_for_port and hello wait after this
        returns.
        """
        if self._proc is not None:
            self._shutdown([self._proc])
        self._proc = self._launch(self.port, self.app, False)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            self._shutdown([proc])


def _accepting(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False
