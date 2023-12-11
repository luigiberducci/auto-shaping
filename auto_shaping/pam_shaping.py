"""
Policy-Assessment Metric from:
"Hierarchical Potential-based Reward Shaping from Task Specifications" by Berducci et al., (arxiv, 2021)
https://arxiv.org/abs/2110.02792
"""
import warnings
from typing import Union

import gymnasium
import numpy as np

from shaping.spec.reward_spec import Variable, Constant, RewardSpec
from shaping.utils import monitor_stl_episode
from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.utils.utils import deep_update


class PAMWrapper(CollectionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[tuple[str, float, float]],
        constants: list[tuple[str, float]] = None,
        params: dict = None,
    ):
        self._params = {
            "max_length": None,
        }
        deep_update(self._params, params or {})

        var_names = [var[0] for var in variables]
        super().__init__(env, variables=var_names)

        self._spec = RewardSpec(
            specs=specs,
            variables=variables,
            constants=constants,
        )

        self._stl_safety_specs = []
        self._stl_target_specs = []
        self._stl_comfort_specs = []
        for req_spec in self._spec.specs:
            try:
                if req_spec._operator == "ensure":
                    stl_req = req_spec.to_rtamt()
                    self._stl_safety_specs.append(stl_req)
                elif req_spec._operator in ["achieve", "conquer"]:
                    stl_req = req_spec.to_rtamt()
                    self._stl_target_specs.append(stl_req)
                elif req_spec._operator == "encourage":
                    # encourage is a special case, in which PAM uses average semantics
                    stl_req = req_spec._predicate.to_rtamt()
                    self._stl_comfort_specs.append(stl_req)
                else:
                    raise NotImplementedError()
            except Exception as e:
                warnings.warn(
                    f"Failed to parse requirement: {req_spec}, make sure it is a valid STL formula"
                )

    def _reward(self, obs, done, info):
        reward = 0.0

        if done:
            safety_sat, target_sat, comfort_sat = 1.0, 1.0, 1.0

            for stl_spec in self._stl_safety_specs:
                robustness_trace = monitor_stl_episode(
                    stl_spec,
                    self._variables,
                    self._episode,
                )
                sat = float(robustness_trace[0][1] >= 0)
                safety_sat *= sat

            for stl_spec in self._stl_target_specs:
                robustness_trace = monitor_stl_episode(
                    stl_spec,
                    self._variables,
                    self._episode,
                )
                sat = float(robustness_trace[0][1] >= 0)
                target_sat *= sat

            comfort_sat = 1.0
            comfort_rhos = []
            for stl_spec in self._stl_comfort_specs:
                robustness_trace = monitor_stl_episode(
                    stl_spec,
                    self._variables,
                    self._episode,
                )
                max_len_episode = self._params["max_length"] or len(self._episode)
                robustness_trace = robustness_trace + [
                    [-1, -1] for _ in range((max_len_episode - len(robustness_trace)))
                ]
                rho = np.mean([float(rob >= 0) for t, rob in robustness_trace])
                comfort_rhos.append(rho)

            if len(comfort_rhos) > 0:
                comfort_sat *= float(np.mean(comfort_rhos))

            reward = safety_sat + 0.5 * target_sat + 0.25 * comfort_sat

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
