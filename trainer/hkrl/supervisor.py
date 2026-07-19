"""Keeps a training run alive across individual instance failures.

A wedged instance, a crashed instance process, and a game that launches but
never accepts a connection are all indistinguishable from here: each one
surfaces as a socket error (or, once a worker subprocess dies from an
uncaught one, an EOFError on its pipe -- see the note on RECOVERABLE below).
All three resolve the same way: relaunch that slot, wait for its bridge to
greet a probe connection (see _port_ready for what that does and does not
prove), and rebuild the vec so training continues instead of the whole run
dying with it.

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
    Monitor+DummyVecEnv, where this class's step_async/step_wait -- the only
    place recovery happens -- would no longer be the ones SB3 calls.

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
        wait_for_port: Callable[..., None] = _wait_for_port,
        recover_attempts: int = 3,
        recover_delay: float = 2.0,
        probe_timeout: float = 2.0,
        launch_timeout: float = 120.0,
        ready_timeout: float = 30.0,
        **env_kwargs,
    ):
        self.ports = list(ports)
        self.relaunch = relaunch
        self.wait_for_port = wait_for_port
        self.recover_attempts = recover_attempts
        self.recover_delay = recover_delay
        self.probe_timeout = probe_timeout
        # Two separate budgets, both per attempt: launch_timeout covers cold
        # game start up to the bridge's first accept (a real Hollow Knight
        # launch is tens of seconds, which is why it is far larger than the
        # others), ready_timeout covers accept -> hello. Worst-case time for
        # one abandoned recovery is recover_attempts * len(ports) *
        # (launch_timeout + ready_timeout).
        self.launch_timeout = launch_timeout
        self.ready_timeout = ready_timeout
        self.env_kwargs = env_kwargs
        self._vec = None
        # Reset arguments live here rather than only on the inner vec, which
        # recovery throws away and rebuilds; _apply_pending replays them onto
        # whatever vec is current. Declared before _build_vec because that
        # call reads them.
        self._pending_seed = None
        self._pending_options = None
        self._recovered_async = False
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
        vec = self._live()
        self._recovered_async = False
        self._apply_pending(vec)
        obs = vec.reset()
        self._consume_pending(vec)
        return obs

    # Recovery lives in step_async/step_wait, not in an overridden step():
    # VecEnvWrapper.step() is `step_async(); return self.venv.step_wait()`, so
    # a step() override is bypassed entirely the moment this object is wrapped
    # in VecFrameStack/VecNormalize/VecMonitor -- the configuration it is
    # built for. Both halves are guarded because a broken slot can raise from
    # either, and recovery replaces the vec both were issued against, so an
    # attempt that recovers during step_async must not then wait on the dead
    # vec's in-flight step: _recovered_async carries the fallback result to
    # step_wait instead.

    def step_async(self, actions):
        try:
            self._live().step_async(actions)
        except RECOVERABLE:
            self._recover()
            self._recovered_async = True

    def step_wait(self):
        if self._recovered_async:
            self._recovered_async = False
            return self._recovery_step_result()
        try:
            vec = self._live()
            result = vec.step_wait()
        except RECOVERABLE:
            self._recover()
            return self._recovery_step_result()
        self.reset_infos = list(vec.reset_infos)
        return result

    def seed(self, seed=None):
        # VecEnv.seed only records into self._seeds, but reset() runs on the
        # inner SubprocVecEnv, which reads its own -- so recording alone makes
        # PPO(..., seed=N) a silent no-op. super() supplies the per-env
        # seed+idx expansion, and passing seed back down reproduces the exact
        # same list inside the inner vec (same formula, same num_envs).
        seeds = super().seed(seed)
        self._pending_seed = seeds[0]
        self._live().seed(self._pending_seed)
        return seeds

    def set_options(self, options=None):
        # Same split as seed(): recorded here for the VecEnv contract, pushed
        # down because the inner vec's reset() is what actually reads them.
        super().set_options(options)
        self._pending_options = self._options
        self._live().set_options(self._pending_options)

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
        forced: set = set()
        failure = None
        for attempt in range(1, self.recover_attempts + 1):
            try:
                # Inside the loop, so a raise from _force_close's
                # join/terminate is a failed attempt like any other rather
                # than escaping recovery. Idempotent: a no-op once self._vec
                # is None, which it is from the second attempt on.
                #
                # Runs before the probes because probing a port opens a
                # connection, and the mod's AcceptLoop (mod/BridgeServer.cs)
                # drops its previous client whenever a new one connects, so
                # probing while the old vec still holds connections would
                # break the very slots being tested.
                self._drop_vec()
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
                    self.wait_for_port(port, timeout=self.launch_timeout)
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
                self._apply_pending(vec)
                vec.reset()
                self._consume_pending(vec)
        except Exception:
            _force_close(vec)
            raise
        self._vec = vec

    def _apply_pending(self, vec: SubprocVecEnv) -> None:
        # A rebuilt vec starts with no seed or options, so anything the caller
        # set before the failure has to be replayed onto it before its reset.
        if self._pending_seed is not None:
            vec.seed(self._pending_seed)
        if self._pending_options is not None:
            vec.set_options(self._pending_options)

    def _consume_pending(self, vec: SubprocVecEnv) -> None:
        # VecEnv.reset() is defined to consume seed/options, so they are
        # cleared on both objects at once; reset_infos is copied back up
        # because callers read it off this object, not the inner vec.
        self.reset_infos = list(vec.reset_infos)
        self._reset_seeds()
        self._reset_options()
        self._pending_seed = None
        self._pending_options = None

    def _drop_vec(self) -> None:
        if self._vec is not None:
            vec, self._vec = self._vec, None
            _force_close(vec)

    def _recovery_step_result(self):
        n = len(self.ports)
        obs = np.zeros((n,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.ones(n, dtype=bool)
        # terminal_observation is part of SB3's done-step contract: VecFrameStack
        # (and anything else that resets per-env buffers on done) reads it, and
        # warns when a wrapped vec omits it.
        infos = [{"terminal_observation": obs[i]} for i in range(n)]
        return obs, rewards, dones, infos


def _port_ready(port: int, timeout: float, host: str = "127.0.0.1") -> bool:
    """True if the bridge's accept thread is alive enough to greet a client.

    This is a bridge-thread probe, NOT a game-liveness probe. In
    mod/BridgeServer.cs the same background AcceptLoop that accepts also
    writes `hello` inline under `gate`, before any game code runs, so the
    line proves only that the process exists, the listener is up, and `gate`
    is free. It says nothing about the Unity main thread: an instance whose
    LateUpdate has stopped calling ReadMessage/SendState still greets this
    probe and still fails every subsequent step.

    What it does catch: a dead/absent process (no connect), a bridge that has
    not finished starting (connect but no hello), and the gate-stalled
    variant where an unbounded SendState write to a peer that stopped
    draining blocks AcceptLoop (connect, hello never arrives).

    A truthful main-thread probe is not reachable from here. The only wire
    message the main thread answers is `reset` (EpisodeManager.LateUpdate);
    driving one costs a full ResetMacro cycle (ResetMacroBudgetSeconds is
    22.5s) and leaves the game mid-reset when the probe disconnects, so the
    real reset that follows has to unwind a live fight. A cheap probe would
    need a no-op ping the mod answers from LateUpdate, which is a mod-side
    protocol change. Until then the frozen-main-thread case is detected one
    layer up: _build_vec's reset fails, and _recover's generic handler
    relaunches every slot because the failing one is not identifiable.
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
