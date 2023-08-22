import gymnasium

from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.spec.reward_spec import RewardSpec
from shaping.utils.utils import monitor_stl_episode


class TLTLWrapper(CollectionWrapper):
    """
    Robustness-based reward shaping from:
    "Reinforcement Learning With Temporal Logic Rewards" by Li, Vasile, and Belta (2016)
    https://arxiv.org/abs/1612.03471
    """

    def __init__(
        self,
        env: gymnasium.Env,
        spec: RewardSpec,
    ):
        super(TLTLWrapper, self).__init__(
            env,
            spec.variables,
        )
        self._spec = spec

    def _reward(self, obs, done, info):
        reward = 0.0

        if done:
            joint_spec = " and ".join(self._spec.specs)
            robustness_trace = monitor_stl_episode(
                joint_spec,
                self._spec.variables,
                self._episode,
            )
            reward = robustness_trace[0][1]

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
