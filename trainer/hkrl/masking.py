"""Keep async-reset placeholder transitions out of the PPO gradient.

AsyncResetWrapper's "isolated" mode fills reset downtime with placeholder
transitions: all-zero observations, zero reward, reset_pending=True in info
(hkrl/async_reset.py). RealEpisodeVecMonitor keeps them out of the metrics,
but they still enter the rollout buffer -- roughly a fifth of the steps a
two-instance run collects -- where the policy and value heads spend real
gradient learning fake zero-reward episodes. Measured against an N=1
control this drags learning below parity even at equal REAL timesteps.

RecurrentPPO already excludes sequence padding from every loss term (and
from advantage normalization) through the per-sample mask its buffer emits,
so the fix rides that mechanism: the buffer tracks which rows are real, the
model zeroes the rows a placeholder step is about to occupy, and
_get_samples multiplies the validity into the padding mask. Minibatches
left with fewer than two real steps are dropped entirely -- masked means
over an empty selection and the advantage std over a single sample are both
NaN, which would poison the optimizer -- and a reset window of ~150
consecutive placeholder steps easily swallows whole minibatches.
"""
import numpy as np

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer


class MaskedRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer with a per-row validity flag folded into the
    loss mask."""

    def reset(self):
        super().reset()
        self.valid = np.ones((self.buffer_size, self.n_envs),
                             dtype=np.float32)

    def get(self, batch_size=None):
        if not self.generator_ready:
            # The parent's first get() flattens every stored tensor from
            # (n_steps, n_envs) order; valid has to follow along or its
            # rows no longer line up with batch_inds.
            self.valid = self.swap_and_flatten(self.valid)
        for samples in super().get(batch_size):
            # A reset window is ~150 consecutive placeholder rows, so a
            # minibatch can be left with 0 valid steps (masked means are
            # NaN) or 1 (advantage std over a single sample is NaN); one
            # optimizer step on either spreads NaN to every weight.
            if int((samples.mask > 1e-8).sum()) >= 2:
                yield samples

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        # The parent just built self.pad_and_flatten for exactly this
        # batch; multiplying keeps the padding zeros it already emitted.
        return samples._replace(
            mask=samples.mask * self.pad_and_flatten(self.valid[batch_inds]))


class MaskedRecurrentPPO(RecurrentPPO):
    """RecurrentPPO that trains only on real transitions.

    Placeholder steps still occupy buffer rows (the vec env is lockstep, so
    they cannot be dropped at collection time) and the LSTM still rolls
    through them, but they are bracketed by dones on both sides -- GAE never
    bootstraps a real episode through them -- and the mask keeps them out of
    the policy, value, and entropy losses.
    """

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = MaskedRecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.rollout_buffer.hidden_state_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _update_info_buffer(self, infos, dones=None):
        # collect_rollouts calls this right after env.step() and before
        # rollout_buffer.add(), so self.rollout_buffer.pos is the row this
        # step's transition is about to occupy.
        super()._update_info_buffer(infos, dones)
        for i, info in enumerate(infos):
            if info.get("reset_pending"):
                self.rollout_buffer.valid[self.rollout_buffer.pos, i] = 0.0
