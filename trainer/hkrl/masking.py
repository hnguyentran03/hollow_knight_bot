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
so the exclusion rides that mechanism: the buffer tracks which rows are
real, the model zeroes the rows a placeholder step is about to occupy, and
_get_samples multiplies the validity into the padding mask.

Masking the loss is not enough on its own, though. RecurrentPPO minibatches
are CONTIGUOUS row slices and every loss is a mean over the rows the mask
keeps, so a slice straddling a ~150-400 row placeholder window trains a
full optimizer step on its handful of surviving rows -- and those rows are
exactly the episode boundaries (the steps before a death, the first steps
of a fight), the noisiest advantages in the buffer, upweighted as much as
32x. Two 500k-step N=2 runs showed the resulting signature: reward climbs
to N=1-level peaks, then a boundary-noise kick knocks the policy off, on
repeat, while per-generation weight deltas stay indistinguishable from a
healthy run's. So get() indexes minibatches over REAL rows only: every
optimizer step averages a full batch of real transitions and placeholder
rows never enter a batch at all. Skipping rows is safe for the recurrent
sequencing because the wrapper brackets every placeholder window with
dones -- the row after a skipped window is an episode start, which already
restarts the LSTM sequence -- and stream_breaks() flags the remaining
discontinuities (env-column starts, the rotation junction) that
episode_starts can't know about.
"""
import numpy as np

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer


def stream_breaks(indices, buffer_size):
    """Boolean array: which positions of a sampled row-index stream must
    start a fresh LSTM sequence because the stream is not buffer-contiguous
    there.

    True at the stream start, wherever consecutive indices are not adjacent
    buffer rows (a skipped placeholder window, or the wrap of the rotation
    trick), and at every env column's first row (flattening is env-major,
    so position k*buffer_size belongs to a different env than k*buffer_size
    - 1 despite being adjacent).
    """
    breaks = np.zeros(len(indices), dtype=bool)
    breaks[0] = True
    if len(indices) > 1:
        prev, cur = indices[:-1], indices[1:]
        breaks[1:] = (cur != prev + 1) | (cur % buffer_size == 0)
    return breaks


class MaskedRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer that samples minibatches from real rows only,
    with a per-row validity flag folded into the loss mask as a backstop."""

    def reset(self):
        super().reset()
        self.valid = np.ones((self.buffer_size, self.n_envs),
                             dtype=np.float32)

    def get(self, batch_size=None):
        # The parent's flattening preamble, replicated (sb3-contrib 2.9.0)
        # because the sampling loop below replaces the parent's: valid has
        # to follow the same reordering or its rows no longer line up.
        assert self.full, "Rollout buffer must be full before sampling"
        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi",
                           "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)
            for tensor in ["observations", "actions", "values", "log_probs",
                           "advantages", "returns", "hidden_states_pi",
                           "cell_states_pi", "hidden_states_vf",
                           "cell_states_vf", "episode_starts"]:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.valid = self.swap_and_flatten(self.valid)
            self.generator_ready = True

        total = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = total

        # Minibatches index over real rows only, so every optimizer step
        # averages a full batch of real transitions (see module docstring).
        # The parent's rotation trick keeps sequence order while varying
        # the chunking between epochs.
        valid_idx = np.flatnonzero(self.valid > 0.5)
        split = np.random.randint(len(valid_idx))
        indices = np.concatenate((valid_idx[split:], valid_idx[:split]))

        # (total, 1), matching swap_and_flatten's scalar shape: the
        # sequencer logical_ors this with episode_starts and a shape
        # mismatch would silently broadcast.
        env_change = np.zeros((total, 1), dtype=np.float32)
        env_change[indices[stream_breaks(indices, self.buffer_size)], 0] = 1.0

        start_idx = 0
        while start_idx < len(indices):
            batch_inds = indices[start_idx:start_idx + batch_size]
            # A 1-row tail would make the advantage std NaN.
            if len(batch_inds) >= 2:
                yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        # batch_inds only ever holds valid rows now, so this multiply is a
        # backstop; the padding zeros the parent emitted are kept either
        # way.
        return samples._replace(
            mask=samples.mask * self.pad_and_flatten(self.valid[batch_inds]))


class MaskedRecurrentPPO(RecurrentPPO):
    """RecurrentPPO that trains only on real transitions.

    Placeholder steps still occupy buffer rows (the vec env is lockstep, so
    they cannot be dropped at collection time) and the LSTM still rolls
    through them, but they are bracketed by dones on both sides -- GAE never
    bootstraps a real episode through them -- and the buffer keeps them out
    of every minibatch.
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
