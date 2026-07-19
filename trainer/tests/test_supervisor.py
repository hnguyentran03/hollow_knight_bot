import socket

import pytest

from hkrl.fake_game import FakeGame, obs, state
from hkrl.supervisor import InstanceDown, SupervisedVecEnv


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


def test_a_slot_that_never_comes_back_raises_instance_down():
    port_a, port_b = _free_port(), _free_port()

    def wait_for_port_that_always_times_out(port, timeout=0.3):
        raise TimeoutError(f"port {port} never accepted a connection")

    a = FakeGame([_episode()], port=port_a).__enter__()
    b = FakeGame([_episode()], port=port_b).__enter__()
    try:
        vec = SupervisedVecEnv(
            [port_a, port_b],
            relaunch=lambda slot: None,  # never actually stands anything back up
            wait_for_port=wait_for_port_that_always_times_out,
        )
        try:
            vec.reset()
            a.__exit__(None, None, None)  # instance 0 crashes and stays dead

            with pytest.raises(InstanceDown):
                vec.step([0, 0])
        finally:
            vec.close()  # must tear down cleanly even mid-failed-recovery
    finally:
        b.__exit__(None, None, None)
