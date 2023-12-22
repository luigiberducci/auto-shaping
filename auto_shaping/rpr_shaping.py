"""
Rank-Preserving Reward from:
"Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles" by Veer et al., (ICRA, 2023)
https://ieeexplore.ieee.org/document/10160622
"""
import warnings

import gymnasium
import numpy as np

from auto_shaping.spec.reward_spec import RewardSpec
from auto_shaping.utils import monitor_stl_episode
from auto_shaping.utils.collection_wrapper import CollectionWrapper

from auto_shaping import Variable, Constant
from auto_shaping.utils.utils import extend_state


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def step(x):
    return 1.0 if x > 0 else 0.0


class RPRWrapper(CollectionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[Variable],
        constants: list[Constant] = None,
        params: dict = None,
    ):
        self._params = {
            "base_coeff": 2.01,
            "max_length": None,
        }
        self._params.update(params or {})

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

        self._variables = [var for var in self._spec.variables] + [
            var for var in self._spec.constants
        ]
        extractor_fn = lambda state, action: extend_state(
            env=env, state=state, action=action, spec=self._spec
        )
        super(RPRWrapper, self).__init__(
            env,
            extractor_fn=extractor_fn,
            variables=self._variables,
        )

    def _reward(self, obs, done, info):
        reward = 0.0

        if done:
            joint_safety_spec = " and ".join(
                [f"({spec})" for spec in self._stl_safety_specs]
            )
            joint_target_spec = " and ".join(self._stl_target_specs)

            rhos = []
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
                max_len_episode = self._params["max_length"] or len(self._episode["time"])
                robustness_trace = robustness_trace + [
                    [-1, -1] for _ in range((max_len_episode - len(robustness_trace)))
                ]
                rho = np.mean([float(rob >= 0) for t, rob in robustness_trace]) # this is in 0..1

                # we need to scale rho to be in -1..1
                rho = 2 * rho - 1

                comfort_rhos.append(rho)

            if len(comfort_rhos) > 0:
                comfort_rho = float(np.mean(comfort_rhos))
                rhos.append(comfort_rho)

            # here, we should have 3 rhos for safety, target, comfort respectively
            # we compute rank-preserving reward as in Eq. 6 of the paper
            a = self._params["base_coeff"]
            n = len(rhos)
            exp_coeff = [
                a ** (n - i) for i in range(0, n) if rhos[i] is not None
            ]
            norm_rhos = [step(r) for r in rhos if r is not None]            # step used for sat/unsat
            avg_rhos = [1/n * sigmoid(r) for r in rhos if r is not None]    # sigmoid used to norm in -1..1
            reward = np.dot(exp_coeff, norm_rhos) + np.sum(avg_rhos)

        return reward

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        reward = self._reward(obs, done, info)
        return obs, reward, done, truncated, info
