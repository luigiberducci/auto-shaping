"""
Rank-Preserving Reward from:
"Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles" by Veer et al., (ICRA, 2023)
https://ieeexplore.ieee.org/document/10160622
"""
import warnings

import gymnasium
import numpy as np

from shaping.spec.reward_spec import RewardSpec
from shaping.utils import monitor_stl_episode
from shaping.utils.collection_wrapper import CollectionWrapper
from shaping.utils.utils import deep_update, sigmoid


class RPRWrapper(CollectionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[tuple[str, float, float]],
        constants: list[tuple[str, float]] = None,
        params: dict = None,
    ):
        self._params = {
            "base_coeff": 2.0,
            "scaling_coeff": 1.0,
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
            joint_safety_spec = " and ".join([f"({spec})"for spec in self._stl_safety_specs])
            joint_target_spec = " and ".join(self._stl_target_specs)

            rhos = []
            c = self._params["scaling_coeff"]   # scaling coefficient
            for stl_spec in [joint_safety_spec, joint_target_spec]:
                if stl_spec == "":  # skip if no safety/target specs
                    continue
                robustness_trace = monitor_stl_episode(
                    stl_spec,
                    self._variables,
                    self._episode,
                )
                rho = robustness_trace[0][1]
                rhos.append(rho)

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
                comfort_rho = float(np.mean(comfort_rhos))
                rhos.append(comfort_rho)

            # here, we should have 3 rhos for safety, target, comfort respectively
            # we compute rank-preserving reward as in Eq. 6 of the paper
            a, c = self._params["base_coeff"], self._params["scaling_coeff"]
            n = len(rhos)
            exp_coeff = [a ** (n - i + 1) for i in range(1, n + 1) if rhos[i - 1] is not None]
            norm_rhos = [sigmoid(c * r) + 1/n * r for r in rhos if r is not None]
            reward = np.dot(exp_coeff, norm_rhos)

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
