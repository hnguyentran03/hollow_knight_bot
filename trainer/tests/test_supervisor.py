import socket
import threading

import pytest
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from hkrl.fake_game import FakeGame, obs, state
from hkrl.supervisor import InstanceDown, SupervisedVecEnv

# Every test drives real sockets. These keep a failed recovery bounded in
# wall-clock time without stubbing out any of the code path under test.
FAST = dict(recover_attempts=2, recover_delay=0.0,
            probe_timeout=0.3, ready_timeout=1.0, timeout=0.5)


class WedgedGame:
    """A game that accepts TCP connections and then answers nothing.

    Models the failure the mod's threading makes possible: BridgeServer's
    AcceptLoop runs independently of the game, so a wedged instance still
    completes the TCP handshake while never producing a `hello` or a state.
    """

    def __init__(self, port):
        self._port = port
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


def _episode(steps=50):
    frames = [state(obs()) for _ in range(steps)]
    frames.append(state(obs(), done=True))
    return frames


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
        fresh = FakeGame([_episode()], port=port_a)
        fresh.__enter__()
        spawned.append(fresh)

    a = FakeGame([_episode()], port=port_a).__enter__()
    # Instance 1 needs a second scripted episode: recovery rebuilds the
    # whole vec (SubprocVecEnv can't isolate which slot failed), so the
    # still-alive instance is reset a second time even though it never dies.
    b = FakeGame([_episode(), _episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv([port_a, port_b], relaunch=relaunch)
        try:
            vec.reset()
            a.__exit__(None, None, None)  # instance 0 crashes mid-run

            step_obs, rewards, dones, infos = vec.step([0, 0])
            assert relaunched == [0]
            assert step_obs.shape[0] == 2
            assert dones.tolist() == [True, True]

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
            # take over and step() below would no longer be the recovery one.
            wrapped = BaseAlgorithm._wrap_env(vec, verbose=0)
            assert wrapped is vec
            assert type(wrapped).step is SupervisedVecEnv.step
            assert wrapped.num_envs == 2
        finally:
            vec.close()
    finally:
        a.__exit__(None, None, None)
        b.__exit__(None, None, None)
