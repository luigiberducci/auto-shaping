"""
Bounded Robustness-based reward shaping from:
"Structured Reward Shaping using Signal Temporal Logic specifications" by Balakrishnan, and Deshmukh (IROS, 2019)

https://ieeexplore.ieee.org/document/8968254
"""

import warnings

import gymnasium

from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.spec.reward_spec import RewardSpec
from shaping.utils.utils import monitor_stl_episode, deep_update


class BHNRWrapper(CollectionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[tuple[str, float, float]],
        constants: list[tuple[str, float]] = None,
        params: dict[str, float] = None,
    ):
        self._params = {
            "window_length": 10,
        }
        deep_update(self._params, params or {})

        var_names = [var[0] for var in variables]
        super().__init__(env, variables=var_names, window_len=self._params["window_length"])

        self._spec = RewardSpec(
            specs=specs,
            variables=variables,
            constants=constants,
        )

        reqs = []
        for req_spec in self._spec.specs:
            try:
                stl_req = req_spec.to_rtamt()
                reqs.append(stl_req)
            except Exception as e:
                warnings.warn(
                    f"Failed to parse requirement: {req_spec}, make sure it is a valid STL formula"
                )
        self._stl_spec = " and ".join(reqs)
        self._variables = list(self._spec.variables.keys())

        super(BHNRWrapper, self).__init__(
            env,
            self._variables,
        )

    def _reward(self, obs, done, info):
        reward = 0.0

        if done:
            robustness_trace = monitor_stl_episode(
                self._stl_spec,
                self._variables,
                self._episode,
            )
            reward = robustness_trace[0][1]

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
