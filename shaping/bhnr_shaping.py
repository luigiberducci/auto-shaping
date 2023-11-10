import warnings

import gymnasium

from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.spec.reward_spec import RewardSpec, Variable
from shaping.utils.utils import monitor_stl_episode, monitor_filtering_stl_episode


class BHNRWrapper(CollectionWrapper):
    """
    Robustness-based reward shaping from:
    "Structured Reward Shaping using Signal Temporal Logic specification" by Balakrishnan, and Deshmukh (2019)
    https://ieeexplore.ieee.org/document/8968254
    """

    def __init__(
            self,
            env: gymnasium.Env,
            specs: list[str],
            variables: list[Variable],
            constants: list[tuple[str, float]] = None,
            window_len: int = 10,
    ):
        var_names = [var[0] for var in variables]
        super().__init__(env, variables=var_names)

        self._spec = RewardSpec(specs=specs, variables=variables, constants=constants, )

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
            env, self._variables,
            window_len=window_len,
        )

    def _reward(self, obs, done, info):
        reward = 0.0

        # compute robustness if episode is at least 2 steps long
        if len(self._episode["time"]) > 1:
            robustness_trace = monitor_filtering_stl_episode(
                self._stl_spec, self._variables, self._episode,
            )
            reward = robustness_trace[0][1]

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
