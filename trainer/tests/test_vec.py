from hkrl.fake_game import FakeGame, obs, state
from hkrl.vec import make_vec


def _episode(steps=3):
    frames = [state(obs()) for _ in range(steps)]
    frames.append(state(obs(), done=True))
    return frames


def test_make_vec_drives_every_instance():
    with FakeGame([_episode()]) as a, FakeGame([_episode()]) as b:
        vec = make_vec([a.port, b.port])
        try:
            obs = vec.reset()
            assert obs.shape[0] == 2
            obs, rewards, dones, infos = vec.step([0, 0])
            assert obs.shape[0] == 2
            assert len(rewards) == 2
        finally:
            vec.close()
