"""Owns the training run's game processes: launch, relaunch, shut down.

train.py runs the games as its own children rather than beside a separately
started launcher: the supervisor's relaunch(slot) callback has to terminate
and reap the game currently holding the port, and a process can only reap
its own children. A relaunch driven from outside the launching process would
orphan a game that launcher's shutdown() never sees, leaving a zombie bound
to the port.

GameProcess is the single-instance unit; GameFleet composes N of them, one
per port, and presents the exact (ports, relaunch(slot)) surface
SupervisedVecEnv is built around.
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
                 launch_timeout: float = 120.0,
                 # Run between terminating the old holder and launching the
                 # replacement -- the relaunch path's chance to fix what a
                 # plain re-exec cannot (train.py wires the clone-save
                 # re-seed here: the wrong-save boot flake means the SAVE
                 # can be the broken part, and rebooting into it re-flakes).
                 prepare: Callable | None = None,
                 # Session-scoped like train.py's --auto: a headless run
                 # relaunches headless, but nothing records the choice.
                 headless: bool = False,
                 # Session-scoped like headless: recorded by train.py's
                 # config, so a resume inherits it there, not here.
                 timescale: float = 1.0):
        self.port = port
        self.app = Path(app)
        self.launch_timeout = launch_timeout
        self._launch = launch
        self._shutdown = shutdown
        self._wait_for_port = wait_for_port
        self._prepare = prepare
        self.headless = headless
        self.timescale = timescale
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        """Launch the game and wait for its bridge to accept."""
        self.spawn()
        self.wait_ready()

    def spawn(self) -> None:
        """Launch the game without waiting for its bridge.

        Split from wait_ready() so GameFleet can boot every instance in
        parallel (a cold Hollow Knight launch is tens of seconds; booting N
        games serially would multiply that) and only then wait for each
        bridge in turn.

        Fails fast on a squatter: this process can only reap its own
        children, so a port held by a leftover game from an earlier run
        would make every relaunch collide (the mod's TcpListener.Start()
        throws on an in-use port) while the readiness probe happily greets
        the zombie.
        """
        if _accepting(self.port):
            if sys.platform == "win32":
                find_it = (f"`netstat -ano | findstr :{self.port}` then "
                           f"`taskkill /PID <pid> /F`")
            else:
                find_it = f"`lsof -nP -iTCP:{self.port} -sTCP:LISTEN`"
            raise PortInUse(
                f"port {self.port} is already accepting connections before "
                f"any game was launched -- an unmanaged process (a leftover "
                f"from an earlier run?) holds it. Find it with "
                f"{find_it}, kill it, rerun."
            )
        self._proc = self._launch(self.port, self.app, False, self.headless,
                                  self.timescale)

    def wait_ready(self) -> None:
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
        if self._prepare is not None:
            self._prepare()
        self._proc = self._launch(self.port, self.app, False, self.headless,
                                  self.timescale)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            self._shutdown([proc])


class GameFleet:
    """N game processes, one per port, behind one training run.

    Explicit ports rather than a base+count so the caller decides the
    numbering (train.py hands out consecutive ports from --port) and so a
    port list can skip one that something else on the machine occupies.

    Every per-instance failure mode stays GameProcess's: the fleet only adds
    the all-or-nothing start (no half-started fleet survives a collision or
    a launch timeout) and the slot -> process routing for the supervisor's
    relaunch callback.

    `apps`, when given, is one game binary per port (see
    launch_instances.prepare_instance): each slot launches -- and, crucially,
    RELAUNCHES -- its own isolated app clone, so a recovery never points a
    replacement game at the master save directory.
    """

    def __init__(self, ports, app: Path = None, apps=None, prepares=None, **process_kwargs):
        ports = list(ports)
        if apps is not None and len(apps) != len(ports):
            raise ValueError("apps must match ports one to one")
        if apps is None:
            apps = [app] * len(ports)
        if prepares is not None and len(prepares) != len(ports):
            raise ValueError("prepares must match ports one to one")
        if prepares is None:
            prepares = [None] * len(ports)
        self.games = []
        for p, a, pr in zip(ports, apps, prepares):
            kwargs = dict(process_kwargs)
            if a is not None:
                kwargs["app"] = a
            kwargs["prepare"] = pr
            self.games.append(GameProcess(port=p, **kwargs))

    @property
    def ports(self) -> list:
        return [g.port for g in self.games]

    def start(self) -> None:
        """Spawn every instance, then wait for every bridge.

        Spawn-all-then-wait-all so the games boot in parallel. Any failure
        -- a squatted port mid-spawn, a bridge that never comes up -- stops
        whatever was already launched before propagating: a partial fleet
        left running would squat its own ports and turn the next start()
        into the PortInUse it exists to diagnose.
        """
        try:
            for g in self.games:
                g.spawn()
            for g in self.games:
                g.wait_ready()
        except Exception:
            self.stop()
            raise

    def relaunch(self, slot: int) -> None:
        """SupervisedVecEnv's relaunch callback: slot indexes self.ports."""
        self.games[slot].relaunch()

    def stop(self) -> None:
        for g in self.games:
            g.stop()


def _accepting(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False
