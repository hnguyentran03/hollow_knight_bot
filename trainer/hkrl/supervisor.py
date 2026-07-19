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


def _log(message: str) -> None:
    """One operator-facing line on stderr.

    stderr and flush=True for the same reasons as
    trainer/scripts/random_agent.py's heartbeats: SB3 writes its progress
    tables to stdout, so recovery lines have to land on the other stream to
    interleave legibly, and an unflushed line is invisible for however long
    the block lasts when the run is piped to a file. The `supervisor:` prefix
    is the grep handle after a twelve-hour run.
    """
    print(f"supervisor: {message}", file=sys.stderr, flush=True)


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
    port -- e.g. trainer/scripts/launch_instances.launch() against the port
    that slot was built with. SupervisedVecEnv itself only decides *which*
    slot needs relaunching and waits for the result; it has no app path to
    launch a replacement with, so that stays the caller's job.

    `relaunch(slot)` must also terminate and reap whatever currently holds
    that port BEFORE starting the replacement -- launch_instances.shutdown()
    on the old Popen, then launch(). This is not optional cleanliness: a
    wedged instance is relaunched precisely because its main thread is dead
    while its bridge still accepts (see _port_ready), so it is still bound to
    the port. Start the replacement without reaping and the new process's
    TcpListener.Start() throws inside the mod on the in-use port, while
    `wait_for_port` returns success immediately against the zombie that is
    still accepting. Recovery then "succeeds" against the dead instance, the
    rebuild's reset fails, every remaining attempt burns the same way, and
    the run ends in InstanceDown with a second unmanaged game process left
    running.
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
        # The fallback step result a step_async-half recovery owes its
        # step_wait, or None when nothing is owed. A result rather than a
        # boolean so "recovered, result not yet delivered" cannot be confused
        # with "a real step is in flight" -- see the pairing checks below.
        self._pending_result = None
        # Observation from the reset _build_vec runs on a rebuilt vec. Written
        # only by _build_vec(reset=True), which runs only inside _recover(),
        # and read only on the branch immediately after a _recover() that
        # returned -- by reset() and by _recovery_step_result() alike. Since
        # _recover() returns only once an attempt has completed that write, and
        # raises otherwise, neither reader can see an earlier rebuild's value.
        self._reset_obs = None
        # Cumulative successful recoveries. Read by the training manifest so
        # an operator sees instability per generation without grepping this
        # module's stderr lines -- a count climbing every generation is the
        # App Nap relaunch-loop signature (an occluded window being suspended
        # over and over).
        self.recoveries = 0
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
        # Guarded like the step halves: PPO.learn() -> _setup_learn() ->
        # env.reset() is the first thing that touches an instance, and
        # wait_for_port returning proves only that the bridge thread bound its
        # listener (see _port_ready), so an instance still loading -- or one
        # that died during load -- fails here more readily than anywhere else.
        # Unguarded, that single slot ends the run at t=0.
        self._require_no_pending("reset")
        try:
            vec = self._live()
            self._apply_pending(vec)
            obs = vec.reset()
        except RECOVERABLE as exc:
            self._recover("reset", exc)
            # _recover ends in a rebuilt, reset vec, so its reset is this
            # reset; re-resetting would burn a second episode for nothing.
            return self._reset_obs
        self._consume_pending(vec)
        return obs

    # Recovery lives in step_async/step_wait, not in an overridden step():
    # VecEnvWrapper.step() is `step_async(); return self.venv.step_wait()`, so
    # a step() override is bypassed entirely the moment this object is wrapped
    # in VecFrameStack/VecNormalize/VecMonitor -- the configuration it is
    # built for. Both halves are guarded because a broken slot can raise from
    # either, and recovery replaces the vec both were issued against, so an
    # attempt that recovers during step_async must not then wait on the dead
    # vec's in-flight step: _pending_result carries the fallback to step_wait.
    #
    # Carrying a result across the halves only works if they alternate, which
    # VecEnv.step and VecEnvWrapper.step both guarantee. _require_no_pending
    # turns the two ways of violating that into an immediate error instead of
    # silent corruption: a second step_async would queue a real step behind
    # the undelivered fallback and leave every later result one step stale,
    # and an interleaved reset would drop the fallback, sending step_wait to
    # recv() on a vec with nothing in flight -- a permanent block, since
    # SubprocVecEnv's pipe reads have no timeout.

    def step_async(self, actions):
        self._require_no_pending("step_async")
        try:
            self._live().step_async(actions)
        except RECOVERABLE as exc:
            self._recover("step_async", exc)
            self._pending_result = self._recovery_step_result()

    def step_wait(self):
        pending, self._pending_result = self._pending_result, None
        if pending is not None:
            return pending
        try:
            vec = self._live()
            result = vec.step_wait()
        except RECOVERABLE as exc:
            self._recover("step_wait", exc)
            return self._recovery_step_result()
        self.reset_infos = list(vec.reset_infos)
        return result

    def _require_no_pending(self, caller: str) -> None:
        if self._pending_result is not None:
            raise RuntimeError(
                f"{caller}() called while a recovery step result from an "
                "earlier step_async is still undelivered; step_async must be "
                "followed by step_wait before anything else"
            )

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

    def _recover(self, caller: str = "?", trigger: BaseException | None = None) -> None:
        # caller/trigger are logging-only: which VecEnv call raised, and the
        # exception that stood in for the failure (see RECOVERABLE -- a dead
        # worker surfaces as a bare EOFError, so the type is often the only
        # thing distinguishing a killed instance from a socket timeout).
        _log(
            f"recovery started from {caller}() "
            f"(detected as {type(trigger).__name__}: {trigger})"
        )
        forced: set = set()
        relaunched: set = set()
        failure = None
        for attempt in range(1, self.recover_attempts + 1):
            _log(f"attempt {attempt}/{self.recover_attempts}")
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
                newly = self._ensure_ready(forced)
                relaunched |= newly
                # A slot just relaunched is booting through multiple reset
                # budgets (see the generic-failure branch below); forcing it
                # again next attempt would restart that boot from the title
                # screen every time, so recovery could never converge. Once
                # relaunched this recovery, a slot is only retried, not
                # re-forced -- until the generic handler re-forces every slot
                # because the failing one is unidentifiable.
                forced -= newly
                self._build_vec(reset=True)
                _log(f"attempt {attempt} succeeded; vec rebuilt and reset, training resumes")
                self.recoveries += 1
                return
            except _RecoveryFailed as exc:
                # A slot that could not be brought back stays forced: its
                # port may now accept connections without the instance behind
                # it being the one this attempt asked for.
                failure = exc
                forced.add(exc.slot)
                _log(f"attempt {attempt} failed: {exc}")
            except Exception as exc:
                # A slot relaunched during this recovery is expected to
                # fail rebuilds for a while: a fresh boot rejoins the
                # fight across several reset budgets (the boot-confirm
                # macro in mod/EpisodeManager.cs), and each budget expiry
                # surfaces here as a generic rebuild failure. Re-forcing
                # it would restart the boot from the title screen every
                # time, so recovery could never converge -- retry as-is
                # and let the boot ratchet forward.
                failure = exc
                if relaunched:
                    _log(
                        f"attempt {attempt} failed: {type(exc).__name__}: {exc}; "
                        f"slots {sorted(relaunched)} were relaunched this recovery "
                        f"and are likely still booting -- retrying without "
                        f"forcing further relaunches"
                    )
                else:
                    # Every slot passed the readiness probe yet the rebuild or
                    # its reset still failed, so at least one instance answers
                    # sockets without being able to serve the protocol. Which
                    # one is not observable from here (SubprocVecEnv
                    # round-trips only slot 0 in its constructor and reports
                    # any other slot's death as an EOFError carrying no
                    # index), so the next attempt relaunches all of them
                    # rather than retrying against a wedged process.
                    forced = set(range(len(self.ports)))
                    _log(
                        f"attempt {attempt} failed after every slot probed ready: "
                        f"{type(exc).__name__}: {exc}; the failing slot is not "
                        f"identifiable, so all {len(self.ports)} will be relaunched"
                    )
            if attempt < self.recover_attempts:
                _log(f"waiting {self.recover_delay}s before the next attempt")
                time.sleep(self.recover_delay)
        _log(
            f"EXHAUSTED after {self.recover_attempts} attempts; raising "
            f"InstanceDown -- the run ends here"
        )
        raise InstanceDown(
            f"recovery abandoned after {self.recover_attempts} attempts; "
            f"last failure: {type(failure).__name__}: {failure}"
        ) from failure

    def _ensure_ready(self, forced: Iterable[int]) -> set:
        forced = set(forced)
        relaunched: set = set()
        for slot, port in enumerate(self.ports):
            if slot in forced:
                reason = "forced by an earlier failed attempt"
            elif not _port_ready(port, self.probe_timeout):
                reason = "failed its readiness probe"
            else:
                _log(f"slot {slot} (port {port}) is ready; not relaunching")
                continue
            _log(f"slot {slot} (port {port}) {reason}; relaunching")
            try:
                self.relaunch(slot)
                self.wait_for_port(port, timeout=self.launch_timeout)
                _log(
                    f"slot {slot} (port {port}) accepted; waiting up to "
                    f"{self.ready_timeout}s for the bridge to say hello"
                )
                _wait_until_ready(port, self.probe_timeout, self.ready_timeout)
                _log(f"slot {slot} (port {port}) back up")
                relaunched.add(slot)
            except Exception as exc:
                raise _RecoveryFailed(
                    slot,
                    f"slot {slot} (port {port}) did not come back up: {exc!r}",
                ) from exc
        return relaunched

    def _build_vec(self, reset: bool) -> None:
        # __new__ then __init__ rather than SubprocVecEnv(...): the
        # constructor starts every worker before it round-trips get_spaces on
        # slot 0, so a slot that cannot serve that round trip raises out of
        # the constructor with the already-started workers unreachable. Split
        # this way, the half-built object -- and its worker handles -- is
        # still in hand to be closed.
        vec = SubprocVecEnv.__new__(SubprocVecEnv)
        if reset:
            # reset=True is only ever the recovery path; construction passes
            # False and stays silent so a normal startup logs nothing.
            _log(f"all slots ready; rebuilding vec over {len(self.ports)} slots")
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
                self._reset_obs = vec.reset()
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
        # The rebuilt vec's own reset observation, the same one reset() hands
        # back after recovering. SB3 stores this as _last_obs, picks the next
        # action from it, and (once wrapped) feeds it to VecNormalize's running
        # statistics and VecFrameStack's buffer, so it has to be a state the
        # game can actually be in.
        obs = self._reset_obs
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.ones(n, dtype=bool)
        # terminal_observation is part of SB3's done-step contract: VecFrameStack
        # (and anything else that resets per-env buffers on done) reads it, and
        # warns when a wrapped vec omits it. Zeros rather than the reset
        # observation: the instance died before reporting the state it ended
        # in, so this one is genuinely unknown, and a placeholder no real
        # observation can equal beats a plausible-looking fabrication.
        # A fresh array per env, never a view of obs: a consumer that
        # normalizes terminal_observation in place must not be able to rewrite
        # the observation this same call returns to the policy.
        infos = [
            {"terminal_observation": np.zeros(self.observation_space.shape, dtype=np.float32)}
            for _ in range(n)
        ]
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
    not finished starting (connect but no hello), the gate-stalled variant
    where an unbounded SendState write to a peer that stopped draining blocks
    AcceptLoop (connect, hello never arrives), and an App-Nap-suspended
    process -- the kernel completes the handshake into the listener's backlog
    even though no thread is scheduled to accept it, so the connect succeeds
    and the hello never comes.

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
