"""Keeps a training run alive across individual instance failures.

A wedged instance, a crashed instance process, and a game that launches but
never accepts a connection are all indistinguishable from here: each one
surfaces as a socket error (or, once a worker subprocess dies from an
uncaught one, an EOFError on its pipe -- see the note on RECOVERABLE below).
All three resolve the same way: relaunch that slot, wait for its port, and
rebuild the vec so training continues instead of the whole run dying with it.
"""
import socket
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    """A slot's instance did not come back up after a relaunch attempt."""


class SupervisedVecEnv:
    """Wraps SubprocVecEnv, relaunching whichever slot's game died.

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
        **env_kwargs,
    ):
        self.ports = list(ports)
        self.relaunch = relaunch
        self.wait_for_port = wait_for_port
        self.env_kwargs = env_kwargs
        self._vec = SubprocVecEnv([make_env(p, **env_kwargs) for p in self.ports])

    def reset(self):
        return self._vec.reset()

    def step(self, actions):
        try:
            return self._vec.step(actions)
        except RECOVERABLE:
            self._recover()
            return self._recovery_step_result()

    def _recover(self):
        # SubprocVecEnv gives no way to tell which slot's pipe broke (the
        # first broken remote aborts step_wait()'s recv loop before the
        # others are even read), so every port is probed directly and only
        # the ones that stopped accepting connections get relaunched.
        for slot, port in enumerate(self.ports):
            if not _port_alive(port):
                try:
                    self.relaunch(slot)
                    self.wait_for_port(port)
                except Exception as exc:
                    raise InstanceDown(
                        f"slot {slot} (port {port}) did not come back up"
                    ) from exc
        _force_close(self._vec)
        self._vec = SubprocVecEnv(
            [make_env(p, **self.env_kwargs) for p in self.ports]
        )
        # Every slot -- including survivors that were never relaunched --
        # gets a brand new subprocess and connection here, so each one is
        # unreset at this point. Reset now rather than leaving it to the
        # caller: a slot's step() must never be the first message a fresh
        # connection receives (FakeGame/the mod both expect reset first),
        # and this also matches the auto-reset contract callers already
        # rely on elsewhere -- step() only ever hands back a fresh
        # post-reset observation after done=True, never a naked one.
        self._vec.reset()

    def _recovery_step_result(self):
        n = len(self.ports)
        obs = np.zeros((n,) + self._vec.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.ones(n, dtype=bool)
        infos = [{} for _ in range(n)]
        return obs, rewards, dones, infos

    def close(self):
        # Not self._vec.close(): if the last recovery raised InstanceDown
        # before finishing its rebuild, self._vec can still be the old,
        # half-broken vec (one dead worker, others still running), and
        # SubprocVecEnv.close() aborts its own send/join loops on that first
        # broken pipe -- see _force_close.
        _force_close(self._vec)

    def __getattr__(self, name):
        return getattr(self._vec, name)


def _port_alive(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _force_close(vec: SubprocVecEnv) -> None:
    """Best-effort close of every worker in `vec`, isolating one dead slot's
    broken pipe from the rest.

    SubprocVecEnv.close() sends "close" to every remote in a single loop and
    stops at the first one that raises (a dead slot's remote is already a
    broken pipe), which would otherwise leave any worker after it in the
    list running forever as an orphaned process.
    """
    vec.waiting = False
    for remote in vec.remotes:
        try:
            remote.send(("close", None))
        except OSError:
            pass
    for process in vec.processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    vec.closed = True
