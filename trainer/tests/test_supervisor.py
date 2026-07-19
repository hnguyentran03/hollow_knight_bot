import json
import socket
import threading

import pytest
from gymnasium.utils import seeding
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from hkrl.fake_game import FakeGame, obs, state
from hkrl.supervisor import InstanceDown, SupervisedVecEnv, _port_ready

# Every test drives real sockets. These keep a failed recovery bounded in
# wall-clock time without stubbing out any of the code path under test.
FAST = dict(recover_attempts=2, recover_delay=0.0, probe_timeout=0.3,
            launch_timeout=1.0, ready_timeout=1.0, timeout=0.5)


class WedgedGame:
    """A game whose bridge thread still accepts while the game is stuck.

    Models both shapes the mod's threading makes possible, because they are
    not equally detectable:

    - hello=False: BridgeServer's AcceptLoop itself is stalled (an unbounded
      SendState write holding `gate`), so a connection is accepted and then
      nothing at all comes back -- not even the greeting.
    - hello=True: the canonical wedge. The Unity main thread is frozen, so
      EpisodeManager.LateUpdate never reads an action or sends a state, but
      AcceptLoop is a separate thread that greets every client inline. This
      one passes _port_ready and can only be caught by the failing rebuild.
    """

    def __init__(self, port, hello=False):
        self._port = port
        self._hello = hello
        self._conns = []

    def __enter__(self):
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", self._port))
        self._srv.listen(4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._srv.close()
        for conn in self._conns:
            conn.close()

    def _run(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self._conns.append(conn)  # held open, deliberately unanswered
            if self._hello:
                try:
                    conn.sendall(
                        json.dumps({"type": "hello", "version": 1}).encode() + b"\n"
                    )
                except OSError:
                    return


def _episode(steps=50, **first_frame):
    """A scripted episode. **first_frame overrides fields of the reset frame,
    which is how a test tells one instance's post-reset state from another's.
    """
    frames = [state(obs(**first_frame))]
    frames += [state(obs()) for _ in range(steps - 1)]
    frames.append(state(obs(), done=True))
    return frames


# Index of khp/9.0 in HornetEnv._flatten's vector. Tests script a distinct khp
# per instance and read this column back to identify which reset produced an
# observation.
KHP = 4


def _free_port():
    """Reserve and release a port so a relaunch callback can rebind to the
    exact same number, the way the real launcher keeps one fixed port per
    instance slot (see hkrl.instances.port_for).
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_a_dead_instance_is_relaunched_and_the_others_keep_running():
    port_a, port_b = _free_port(), _free_port()
    relaunched = []
    spawned = []  # every fresh FakeGame relaunch() stands up on port_a

    def relaunch(slot):
        assert slot == 0
        relaunched.append(slot)
        fresh = FakeGame([_episode(khp=3)], port=port_a)
        fresh.__enter__()
        spawned.append(fresh)

    a = FakeGame([_episode()], port=port_a).__enter__()
    # Instance 1 needs a second scripted episode: recovery rebuilds the
    # whole vec (SubprocVecEnv can't isolate which slot failed), so the
    # still-alive instance is reset a second time even though it never dies.
    b = FakeGame([_episode(), _episode(khp=7)], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch)
        try:
            vec.reset()
            a.__exit__(None, None, None)  # instance 0 crashes mid-run

            step_obs, rewards, dones, infos = vec.step([0, 0])
            assert relaunched == [0]
            assert step_obs.shape[0] == 2
            assert dones.tolist() == [True, True]

            # The recovery frame carries the rebuilt vec's real post-reset
            # observation, not a placeholder: each slot reports the distinct
            # khp its own reset frame was scripted with -- the relaunched
            # instance's and the survivor's, both from this rebuild.
            assert step_obs[:, KHP] == pytest.approx([3 / 9, 7 / 9], abs=1e-6)
            # terminal_observation stays zeros: the instance died before
            # reporting the state it ended in, so that one is genuinely
            # unknown.
            for info in infos:
                assert info["terminal_observation"].tolist() == [0.0] * step_obs.shape[1]

            # Both slots -- the relaunched one and the untouched survivor --
            # must now be producing real steps again, not just the one
            # fallback frame reported for the recovery step itself.
            step_obs, rewards, dones, infos = vec.step([0, 0])
            assert step_obs.shape[0] == 2
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        b.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_an_instance_dead_at_the_first_reset_is_recovered_not_fatal():
    """reset() is where a half-loaded instance fails most readily.

    wait_for_port returns as soon as the bridge thread binds, which says
    nothing about the game being ready to serve a reset, and PPO.learn()'s
    _setup_learn() resets before any step happens.
    """
    port_a, port_b = _free_port(), _free_port()
    spawned = []

    def relaunch(slot):
        assert slot == 0
        spawned.append(FakeGame([_episode(khp=3)], port=port_a).__enter__())

    a = FakeGame([_episode()], port=port_a).__enter__()
    # Two episodes for the survivor: its first reset succeeds, then the
    # rebuild resets every slot again.
    b = FakeGame([_episode(), _episode(khp=7)], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch, **FAST)
        try:
            a.__exit__(None, None, None)  # dies before the run's first reset

            reset_obs = vec.reset()
            assert spawned  # recovery ran from inside reset()
            assert reset_obs.shape == (2, vec.observation_space.shape[0])
            # It is the rebuild's own reset observation: each slot's scripted
            # khp comes back on its own row.
            assert reset_obs[:, KHP] == pytest.approx([3 / 9, 7 / 9], abs=1e-6)

            # The observation is usable, not a stand-in: stepping from it
            # continues normally on both slots.
            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        b.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_a_failure_in_step_async_carries_its_recovery_to_step_wait():
    """The broken-pipe half of recovery: the worker is already gone when
    step_async sends, so the parent raises there and never reaches recv().
    """
    port_a, port_b = _free_port(), _free_port()
    spawned = []

    def relaunch(slot):
        assert slot == 0
        spawned.append(FakeGame([_episode(khp=3)], port=port_a).__enter__())

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode(), _episode(khp=7)], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch, **FAST)
        try:
            vec.reset()
            a.__exit__(None, None, None)
            # Closing the parent's ends makes the very next send raise
            # instead of the recv that every other test hits first.
            for remote in vec._vec.remotes:
                remote.close()

            vec.step_async([0, 0])
            assert spawned  # recovery happened in the step_async half

            # The undelivered fallback pins the halves together: anything
            # other than step_wait next is an error, not a silent drop or a
            # silently stale result.
            with pytest.raises(RuntimeError, match="step_wait"):
                vec.step_async([0, 0])
            with pytest.raises(RuntimeError, match="step_wait"):
                vec.reset()

            step_obs, _, dones, infos = vec.step_wait()
            assert dones.tolist() == [True, True]
            # Recovery from the step_async half hands back the same real
            # post-reset observation the step_wait half does.
            assert step_obs[:, KHP] == pytest.approx([3 / 9, 7 / 9], abs=1e-6)
            assert infos[0]["terminal_observation"].tolist() == [0.0] * step_obs.shape[1]
            # terminal_observation must not alias the returned observation, or
            # a consumer normalizing it in place rewrites what the policy got.
            handed_to_policy = step_obs[0].copy()
            infos[0]["terminal_observation"] += 1.0
            assert step_obs[0].tolist() == handed_to_policy.tolist()

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        b.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_a_slot_that_never_comes_back_raises_instance_down_once_retries_run_out():
    port_a, port_b = _free_port(), _free_port()
    attempts = []

    def wait_for_port_that_always_times_out(port, timeout=0.3):
        raise TimeoutError(f"port {port} never accepted a connection")

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv(
            [port_a, port_b],
            relaunch=attempts.append,  # never actually stands anything back up
            wait_for_port=wait_for_port_that_always_times_out,
            **FAST,
        )
        try:
            vec.reset()
            a.__exit__(None, None, None)  # instance 0 crashes and stays dead

            with pytest.raises(InstanceDown) as excinfo:
                vec.step([0, 0])
            # Every attempt in the budget was spent before giving up, and the
            # message names the slot that actually failed.
            assert attempts.count(0) == FAST["recover_attempts"]
            assert f"slot 0 (port {port_a})" in str(excinfo.value)
        finally:
            vec.close()  # must tear down cleanly even mid-failed-recovery
    finally:
        b.__exit__(None, None, None)


def test_a_wedged_instance_is_relaunched_even_though_its_port_still_accepts():
    port_a, port_b = _free_port(), _free_port()
    spawned = []

    def relaunch(slot):
        assert slot == 0
        wedged.__exit__(None, None, None)  # the wedged process is replaced
        fresh = FakeGame([_episode()], port=port_a).__enter__()
        spawned.append(fresh)

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode(), _episode()], port=port_b).__enter__()
    wedged = None
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch, **FAST)
        try:
            vec.reset()
            # Instance 0 wedges: its game stops serving the protocol while
            # something keeps the listening socket accepting connections.
            a.__exit__(None, None, None)
            wedged = WedgedGame(port_a).__enter__()

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [True, True]
            assert len(spawned) == 1  # a TCP-only probe would have skipped it

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        b.__exit__(None, None, None)
        if wedged is not None and not spawned:
            wedged.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_a_relaunch_that_fails_once_is_retried_rather_than_ending_the_run():
    port_a, port_b = _free_port(), _free_port()
    calls = []
    spawned = []

    def flaky_relaunch(slot):
        calls.append(slot)
        if len(calls) == 1:
            raise RuntimeError("launcher stuttered")
        spawned.append(FakeGame([_episode()], port=port_a).__enter__())

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode(), _episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=flaky_relaunch, **FAST)
        try:
            vec.reset()
            a.__exit__(None, None, None)

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [True, True]
            assert calls == [0, 0]  # first attempt failed, second one worked

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        b.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_a_frozen_main_thread_passes_the_probe_and_is_caught_by_the_rebuild():
    port_a, port_b = _free_port(), _free_port()
    relaunched = []
    live = {}

    def relaunch(slot):
        relaunched.append(slot)
        port = (port_a, port_b)[slot]
        live.pop(slot).__exit__(None, None, None)
        live[slot] = FakeGame([_episode(), _episode()], port=port).__enter__()

    live[0] = FakeGame([_episode()], port=port_a).__enter__()
    live[1] = FakeGame([_episode(), _episode(), _episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch, **FAST)
        try:
            vec.reset()
            # Instance 0's Unity main thread freezes: EpisodeManager stops
            # serving the protocol while BridgeServer's separate accept thread
            # keeps greeting clients.
            live.pop(0).__exit__(None, None, None)
            live[0] = WedgedGame(port_a, hello=True).__enter__()
            # The probe is a bridge-thread check, not a liveness check: this
            # instance answers it and is therefore NOT singled out below.
            assert _port_ready(port_a, 0.3) is True

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [True, True]
            # First attempt relaunched nothing (both slots greeted the probe);
            # its rebuild reset then failed on the frozen slot, and because
            # SubprocVecEnv cannot say which slot died, the second attempt
            # relaunched every slot -- including the healthy survivor.
            assert relaunched == [0, 1]

            _, _, dones, _ = vec.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            vec.close()
    finally:
        for game in live.values():
            game.__exit__(None, None, None)


def test_recovery_still_runs_when_wrapped_in_an_sb3_vec_wrapper():
    """VecEnvWrapper.step() is step_async() + venv.step_wait(), so recovery
    has to live in that pair, not in an overridden step() the wrapper skips.
    """
    port_a, port_b = _free_port(), _free_port()
    spawned = []

    def relaunch(slot):
        assert slot == 0
        spawned.append(FakeGame([_episode()], port=port_a).__enter__())

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode(), _episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch, **FAST)
        stacked = VecFrameStack(vec, n_stack=4)
        try:
            stacked.reset()
            a.__exit__(None, None, None)

            step_obs, _, dones, _ = stacked.step([0, 0])
            assert spawned  # recovery ran through the wrapper's step path
            assert dones.tolist() == [True, True]
            assert step_obs.shape == (2, 4 * vec.observation_space.shape[0])

            step_obs, _, dones, _ = stacked.step([0, 0])
            assert dones.tolist() == [False, False]
        finally:
            stacked.close()
    finally:
        b.__exit__(None, None, None)
        for fg in spawned:
            fg.__exit__(None, None, None)


def test_seed_and_options_reach_the_inner_envs():
    port_a, port_b = _free_port(), _free_port()
    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=lambda slot: None,
                               **FAST)
        try:
            assert vec.seed(123) == [123, 124]
            vec.set_options({"note": "x"})
            vec.reset()

            # Each env really was reset with its own seed: gym.Env.reset seeds
            # self.np_random, so an identically seeded generator draws the
            # same numbers.
            expected = [seeding.np_random(s)[0].integers(0, 10 ** 9)
                        for s in (123, 124)]
            drawn = [rng.integers(0, 10 ** 9)
                     for rng in vec.get_attr("np_random")]
            assert drawn == expected
            # Consumed by that reset, on this object and the inner vec alike.
            assert vec._seeds == [None, None]
            assert vec._vec._seeds == [None, None]
            assert vec._vec._options == [{}, {}]
        finally:
            vec.close()
    finally:
        a.__exit__(None, None, None)
        b.__exit__(None, None, None)


def test_sb3_accepts_it_as_a_vec_env_and_keeps_the_recovery_step():
    port_a, port_b = _free_port(), _free_port()
    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=lambda slot: None,
                               **FAST)
        try:
            assert isinstance(vec, VecEnv)
            # _wrap_env is what PPO(...) runs on whatever it is handed; it
            # must return this object untouched, or Monitor+DummyVecEnv would
            # take over and the stepping pair below would no longer be the
            # recovering one.
            wrapped = BaseAlgorithm._wrap_env(vec, verbose=0)
            assert wrapped is vec
            assert type(wrapped).step_async is SupervisedVecEnv.step_async
            assert type(wrapped).step_wait is SupervisedVecEnv.step_wait
            assert wrapped.num_envs == 2
        finally:
            vec.close()
    finally:
        a.__exit__(None, None, None)
        b.__exit__(None, None, None)
