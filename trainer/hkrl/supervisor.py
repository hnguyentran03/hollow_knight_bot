"""Keeps a training run alive across individual instance failures.

A wedged instance, a crashed instance process, and a game that launches but
never accepts a connection are all indistinguishable from here: each one
surfaces as a socket error (or, once a worker subprocess dies from an
uncaught one, an EOFError on its pipe -- see the note on RECOVERABLE below).
All three resolve the same way: relaunch that slot, wait for it to answer at
the protocol layer, and rebuild the vec so training continues instead of the
whole run dying with it.

Recovery itself can fail, so it is retried: each attempt relaunches whatever
is not answering, rebuilds, and resets, and only once the attempt budget is
spent does InstanceDown end the run.
"""
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hkrl.protocol import ConnectionClosed
from hkrl.vec import make_env

# scripts/ is not a package; reached via a path insert, same convention as
# trainer/tests/test_launcher.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_instances import wait_for_port as _wait_for_port  # noqa: E402

# stable_baselines3's SubprocVecEnv worker (_worker in subproc_vec_env.py)
# only catches EOFError/KeyboardInterrupt around env.step(); any other
# exception -- socket.timeout, ConnectionClosed, a plain OSError such as
# ConnectionResetError/BrokenPipeError -- kills that worker process outright.
# The main process then observes this not as the original exception but as
# an EOFError reading the dead worker's pipe, so EOFError has to be treated
# as recoverable too.
RECOVERABLE = (socket.timeout, ConnectionClosed, OSError, EOFError)


class InstanceDown(Exception):
    """Recovery ran out of attempts; the run cannot continue."""


class _RecoveryFailed(Exception):
    """A named slot could not be brought back. Never escapes _recover()."""

    def __init__(self, slot: int, message: str):
        super().__init__(message)
        self.slot = slot


class SupervisedVecEnv(VecEnv):
    """Wraps SubprocVecEnv, relaunching whichever slot's game died.

    A VecEnv subclass, not a plain wrapper: BaseAlgorithm._wrap_env checks
    `isinstance(env, VecEnv)` and otherwise buries the object under
    Monitor+DummyVecEnv, where this class's step() -- the only place recovery
    happens -- would no longer be the one SB3 calls.

    `relaunch(slot)` must synchronously (or before `wait_for_port` gives up)
    cause a fresh game process to start listening on that slot's original
    port -- e.g. trainer/scripts/launch_instances.launch() against
    hkrl.instances.port_for(slot). SupervisedVecEnv itself only decides
    *which* slot needs relaunching and waits for the result; it has no home
    directory or app path to launch a replacement with, so that stays the
    caller's job.
    """

    def __init__(
        self,
        ports: Sequence[int],
        relaunch: Callable[[int], None],
        wait_for_port: Callable[[int], None] = _wait_for_port,
        recover_attempts: int = 3,
        recover_delay: float = 2.0,
        probe_timeout: float = 2.0,
        ready_timeout: float = 30.0,
        **env_kwargs,
    ):
        self.ports = list(ports)
        self.relaunch = relaunch
        self.wait_for_port = wait_for_port
        self.recover_attempts = recover_attempts
        self.recover_delay = recover_delay
        self.probe_timeout = probe_timeout
        self.ready_timeout = ready_timeout
        self.env_kwargs = env_kwargs
        self._vec = None
        # No reset here: construction only opens the connections, leaving the
        # first reset to the caller the way a plain SubprocVecEnv does.
        self._build_vec(reset=False)
        super().__init__(
            len(self.ports),
            self._vec.observation_space,
            self._vec.action_space,
        )

    # -- VecEnv API --

    def reset(self):
        return self._live().reset()

    def step(self, actions):
        # Overrides VecEnv.step (step_async + step_wait) rather than
        # implementing recovery in step_wait: a broken slot can raise from
        # either half, and recovery replaces the vec both were issued
        # against, so the pair has to be recovered as one unit.
        try:
            return self._live().step(actions)
        except RECOVERABLE:
            self._recover()
            return self._recovery_step_result()

    def step_async(self, actions):
        self._live().step_async(actions)

    def step_wait(self):
        return self._live().step_wait()

    def get_attr(self, attr_name, indices=None):
        return self._live().get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self._live().set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._live().env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None):
        return self._live().env_is_wrapped(wrapper_class, indices)

    def get_images(self):
        return self._live().get_images()

    def close(self):
        # Not self._vec.close(): if the last recovery raised InstanceDown
        # before finishing its rebuild, self._vec can still be the old,
        # half-broken vec (one dead worker, others still running), and
        # SubprocVecEnv.close() aborts its own send/join loops on that first
        # broken pipe -- see _force_close.
        self._drop_vec()

    def __getattr__(self, name):
        # __dict__ lookup, not self._vec: during __init__ and after a failed
        # recovery there is no vec to forward to, and self._vec would recurse
        # back into here.
        vec = self.__dict__.get("_vec")
        if vec is None:
            raise AttributeError(name)
        return getattr(vec, name)

    # -- recovery --

    def _live(self) -> SubprocVecEnv:
        if self._vec is None:
            raise InstanceDown("no live vec: recovery was abandoned")
        return self._vec

    def _recover(self) -> None:
        # The old vec is torn down first: probing a port opens a connection,
        # and the mod's AcceptLoop (mod/BridgeServer.cs) drops its previous
        # client whenever a new one connects, so probing while the old vec
        # still holds connections would break the very slots being tested.
        self._drop_vec()
        forced: set = set()
        failure = None
        for attempt in range(1, self.recover_attempts + 1):
            try:
                self._ensure_ready(forced)
                self._build_vec(reset=True)
                return
            except _RecoveryFailed as exc:
                # A slot that could not be brought back stays forced: its
                # port may now accept connections without the instance behind
                # it being the one this attempt asked for.
                failure = exc
                forced.add(exc.slot)
            except Exception as exc:
                # Every slot passed the readiness probe yet the rebuild or its
                # reset still failed, so at least one instance answers sockets
                # without being able to serve the protocol. Which one is not
                # observable from here (SubprocVecEnv round-trips only slot 0
                # in its constructor and reports any other slot's death as an
                # EOFError carrying no index), so the next attempt relaunches
                # all of them rather than retrying against a wedged process.
                failure = exc
                forced = set(range(len(self.ports)))
            if attempt < self.recover_attempts:
                time.sleep(self.recover_delay)
        raise InstanceDown(
            f"recovery abandoned after {self.recover_attempts} attempts; "
            f"last failure: {type(failure).__name__}: {failure}"
        ) from failure

    def _ensure_ready(self, forced: Iterable[int]) -> None:
        forced = set(forced)
        for slot, port in enumerate(self.ports):
            if slot in forced or not _port_ready(port, self.probe_timeout):
                try:
                    self.relaunch(slot)
                    self.wait_for_port(port)
                    _wait_until_ready(port, self.probe_timeout, self.ready_timeout)
                except Exception as exc:
                    raise _RecoveryFailed(
                        slot,
                        f"slot {slot} (port {port}) did not come back up: {exc!r}",
                    ) from exc

    def _build_vec(self, reset: bool) -> None:
        # __new__ then __init__ rather than SubprocVecEnv(...): the
        # constructor starts every worker before it round-trips get_spaces on
        # slot 0, so a slot that cannot serve that round trip raises out of
        # the constructor with the already-started workers unreachable. Split
        # this way, the half-built object -- and its worker handles -- is
        # still in hand to be closed.
        vec = SubprocVecEnv.__new__(SubprocVecEnv)
        try:
            vec.__init__([make_env(p, **self.env_kwargs) for p in self.ports])
            # On the recovery path every slot -- including survivors that were
            # never relaunched -- gets a brand new subprocess and connection
            # here, so each one is unreset at this point. Reset here rather
            # than in the caller: a slot's step() must never be the first
            # message a fresh
            # connection receives (FakeGame/the mod both expect reset first),
            # and this also matches the auto-reset contract callers already
            # rely on elsewhere -- step() only ever hands back a fresh
            # post-reset observation after done=True, never a naked one. It
            # is also the only check that reaches every slot: the constructor
            # verifies slot 0 alone.
            if reset:
                vec.reset()
        except Exception:
            _force_close(vec)
            raise
        self._vec = vec

    def _drop_vec(self) -> None:
        if self._vec is not None:
            vec, self._vec = self._vec, None
            _force_close(vec)

    def _recovery_step_result(self):
        n = len(self.ports)
        obs = np.zeros((n,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.ones(n, dtype=bool)
        infos = [{} for _ in range(n)]
        return obs, rewards, dones, infos


def _port_ready(port: int, timeout: float, host: str = "127.0.0.1") -> bool:
    """True only if the instance answers at the protocol layer.

    A TCP connect alone cannot tell a healthy instance from a wedged one: the
    mod's AcceptLoop runs on its own thread and keeps accepting no matter what
    the game is stuck on. The bridge sends its `hello` line immediately on
    accept, so requiring that line is what separates the two.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            with sock.makefile("rb") as f:
                return bool(f.readline())
    except OSError:
        return False


def _wait_until_ready(port: int, probe_timeout: float, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if _port_ready(port, probe_timeout):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"port {port} accepted connections but never sent hello "
                f"within {timeout}s"
            )
        time.sleep(0.5)


def _force_close(vec: SubprocVecEnv) -> None:
    """Best-effort close of every worker in `vec`, isolating one dead slot's
    broken pipe from the rest.

    SubprocVecEnv.close() sends "close" to every remote in a single loop and
    stops at the first one that raises (a dead slot's remote is already a
    broken pipe), which would otherwise leave any worker after it in the
    list running forever as an orphaned process.

    Attributes are read defensively: a vec whose __init__ raised partway is
    passed here too, and may have started workers without having reached
    every attribute a fully built one has.
    """
    vec.waiting = False
    for remote in getattr(vec, "remotes", ()):
        try:
            remote.send(("close", None))
        except OSError:
            pass
    for process in getattr(vec, "processes", ()):
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    vec.closed = True
