import warnings

import gymnasium

from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.spec.reward_spec import RewardSpec, Variable, Constant
from shaping.utils.utils import monitor_stl_episode, extend_state


class TLTLWrapper(CollectionWrapper):
    """
    Robustness-based reward shaping from:
    "Reinforcement Learning With Temporal Logic Rewards" by Li, Vasile, and Belta (2016)
    https://arxiv.org/abs/1612.03471
    """

    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[Variable],
        constants: list[Constant] = None,
    ):
        self._spec = RewardSpec(specs=specs, variables=variables, constants=constants,)

        reqs = []
        for req_spec in self._spec.specs:
            try:
                stl_req = req_spec.to_rtamt()
                reqs.append(stl_req)
            except Exception as e:
                warnings.warn(
                    f"Failed to parse requirement: {req_spec}, if comfort requirement no worries because it is not supported by STL"
                )
        self._stl_spec = " and ".join(reqs)
        self._variables = [var for var in self._spec.variables] + [var for var in self._spec.constants]

        extractor_fn = lambda state: extend_state(state, self._spec)
        super(TLTLWrapper, self).__init__(
            env, extractor_fn=extractor_fn, variables=self._variables,
        )

    def _reward(self, obs, done, info):
        reward = 0.0

        if done:
            robustness_trace = monitor_stl_episode(
                self._stl_spec, self._variables, self._episode,
            )
            reward = robustness_trace[0][1]

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
