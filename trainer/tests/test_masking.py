import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hkrl.fake_slow_env import make_async_timed
from hkrl.masking import MaskedRecurrentPPO, MaskedRecurrentRolloutBuffer

RP = {"reset_pending": True}


class _ScriptedEnv(gym.Env):
    """Replays a fixed cyclic (reward, done, info) script, one per step."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,),
                                   dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        reward, done, info = self._script[self._i % len(self._script)]
        self._i += 1
        return np.zeros(4, dtype=np.float32), reward, done, False, dict(info)


def _make_model(script, n_steps, batch_size):
    venv = DummyVecEnv([lambda: _ScriptedEnv(script)])
    return MaskedRecurrentPPO(
        "MlpLstmPolicy", venv, n_steps=n_steps, batch_size=batch_size,
        n_epochs=1, seed=1, verbose=0, device="cpu",
        policy_kwargs=dict(lstm_hidden_size=8,
                           net_arch=dict(pi=[8], vf=[8])))


def test_masked_model_uses_masked_buffer():
    model = _make_model([(0.0, False, {})], n_steps=8, batch_size=8)
    assert isinstance(model.rollout_buffer, MaskedRecurrentRolloutBuffer)


def test_placeholder_transitions_are_masked_out_of_minibatches():
    """Steps whose info carries reset_pending must contribute nothing to the
    loss mask RecurrentPPO trains with, while every real step still does."""
    script = [
        (0.5, False, {}),
        (1.0, True, {}),    # real episode ends
        (0.0, False, RP),
        (0.0, False, RP),
        (0.0, True, RP),    # throwaway reset-window episode ends
        (0.3, False, {}),
        (0.2, False, {}),
        (0.8, True, {}),    # real episode ends
        (0.0, False, RP),
        (0.0, True, RP),    # second throwaway episode ends
        (0.1, False, {}),
        (0.4, False, {}),
        (0.2, False, {}),
        (0.6, False, {}),
        (0.3, False, {}),
        (0.5, False, {}),
    ]
    junk = {2, 3, 4, 8, 9}
    model = _make_model(script, n_steps=16, batch_size=8)
    model.learn(total_timesteps=16)

    masked_in = sum(int((batch.mask > 1e-8).sum())
                    for batch in model.rollout_buffer.get(8))
    assert masked_in == 16 - len(junk)

    valid = np.asarray(model.rollout_buffer.valid).reshape(-1)
    expected = np.array([0.0 if i in junk else 1.0 for i in range(16)])
    assert np.array_equal(valid, expected)


def test_all_placeholder_minibatch_does_not_poison_the_optimizer():
    """A reset window can span an entire minibatch; a fully-masked batch
    makes every masked mean NaN, which one optimizer step would spread to
    the whole policy. A junk run of 2*batch_size-1 guarantees at least one
    all-junk minibatch whatever split point the buffer shuffles with."""
    script = [(1.0, True, {})]
    script += [(0.0, False, RP)] * 6
    script += [(0.0, True, RP)]     # 7 consecutive junk steps: rows 1-7
    script += [
        (0.5, False, {}),
        (0.2, False, {}),
        (0.7, True, {}),
        (0.1, False, {}),
        (0.3, False, {}),
        (0.2, False, {}),
        (0.4, False, {}),
        (0.6, False, {}),
    ]
    model = _make_model(script, n_steps=16, batch_size=4)
    model.learn(total_timesteps=16)

    for param in model.policy.parameters():
        assert np.isfinite(param.detach().numpy()).all()


def test_masking_sees_reset_pending_through_the_real_stack():
    """The real AsyncResetWrapper's placeholder flag must survive
    VecNormalize's step_wait into the buffer's validity flags, with real
    steps still counted valid."""
    venv = DummyVecEnv([lambda: make_async_timed(reset_s=0.05, episode_len=5,
                                                 pending_mode="isolated")])
    env = VecNormalize(venv, gamma=0.995)
    model = MaskedRecurrentPPO(
        "MlpLstmPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=1,
        verbose=0, device="cpu",
        policy_kwargs=dict(lstm_hidden_size=8,
                           net_arch=dict(pi=[8], vf=[8])))
    model.learn(total_timesteps=64)

    valid = np.asarray(model.rollout_buffer.valid)
    assert (valid == 0.0).any()
    assert (valid == 1.0).any()
