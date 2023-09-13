"""
Hierarchical Potential-based Reward Shaping from:
"Hierarchical Potential-based Reward Shaping from Task Specifications" by Berducci et al., (arxiv, 2021)
https://arxiv.org/abs/2110.02792
"""

from typing import Union

import gymnasium
import numpy as np

from shaping import RewardSpec
from shaping.spec.reward_spec import Variable, Constant
from shaping.utils.utils import clip_and_norm, deep_update

_cmp_lambdas = {
    "<": lambda x, y: x < y,
    ">": lambda x, y: x > y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "==": lambda x, y: (x - y) < 1e-6,
    "!=": lambda x, y: (x - y) > 1e-6,
}


def eval_var(state, var, fn):
    if fn == "abs":
        val = np.abs(state[var])
    elif fn == "exp":
        val = np.exp(state[var])
    elif fn is None:
        val = state[var]
    else:
        raise NotImplementedError(f"Function {fn} not implemented")
    return val


class SparseSuccessRewardWrapper(gymnasium.Wrapper):
    """
    This wrapper provides a reward of 1.0 every time the agent satisfies the target condition.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[
            Union[Variable, tuple[str, float, float], tuple[str, float, float, str]]
        ],
        constants: list[
            Union[Constant, tuple[str, float], tuple[str, float, str]]
        ] = None,
    ):
        super(SparseSuccessRewardWrapper, self).__init__(env)

        self._spec = RewardSpec(specs, variables, constants)
        self._target_specs = [
            spec
            for spec in self._spec.specs
            if spec._operator in ["achieve", "conquer"]
        ]
        self._variables = self._spec.variables

        assert (
            type(env.observation_space) == gymnasium.spaces.Dict
        ), "Observation space must be a dictionary, use DictWrapper"

    def _reward(self, state, action, next_state, done, info):
        reward = 0.0
        for target_spec in self._target_specs:
            (fn, var), aritm_op, threshold = target_spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = eval_var(next_state, var, fn)

            reward += float(cmp_lambda(val, threshold))
        return reward

    def step(self, action):
        next_obs, _, done, truncated, info = super().step(action)
        reward = self._reward(None, action, next_obs, done, info)
        return next_obs, reward, done, truncated, info


class HPRSWrapper(SparseSuccessRewardWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[tuple[str, float, float]],
        constants: list[tuple[str, float]] = None,
        params: dict[str, float] = None,
    ):
        self._params = {
            "gamma": 1.00,
        }
        deep_update(self._params, params or {})

        gymnasium.utils.RecordConstructorArgs.__init__(
            self, specs=specs, variables=variables, constants=constants, gamma=self._params["gamma"]
        )
        super(HPRSWrapper, self).__init__(env, specs, variables, constants)

        # safety, target, comfort
        self._safety_specs = [
            spec for spec in self._spec.specs if spec._operator in ["ensure"]
        ]
        self._target_spec = [
            spec
            for spec in self._spec.specs
            if spec._operator in ["achieve", "conquer"]
        ]
        self._comfort_specs = [
            spec for spec in self._spec.specs if spec._operator in ["encourage"]
        ]

        assert (
            len(self._target_spec) == 1
        ), f"There should be exactly one target specification, got {len(self._target_spec)}"

        self._obs = None

    def reset(self, **kwargs):
        self._obs, info = super().reset(**kwargs)
        return self._obs, info

    def step(self, action):
        next_obs, _, done, truncated, info = super().step(action)
        reward = self._hprs_reward(self._obs, action, next_obs, done, info)
        return next_obs, reward, done, truncated, info

    def _hprs_reward(self, state, action, next_state, done, info):
        base_reward = super()._reward(state, action, next_state, done, info)

        # return base_reward for terminal states
        if done:
            return base_reward

        # hierarchical shaping
        gamma = self._params["gamma"]
        safety_shaping = gamma * self._safety_shaping(
            next_state
        ) - self._safety_shaping(state)
        target_shaping = gamma * self._target_shaping(
            next_state
        ) - self._target_shaping(state)
        comfort_shaping = gamma * self._comfort_shaping(
            next_state
        ) - self._comfort_shaping(state)

        return base_reward + safety_shaping + target_shaping + comfort_shaping

    def _safety_shaping(self, state) -> float:
        reward = 0.0

        for spec in self._safety_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = eval_var(state, var, fn)

            reward += float(cmp_lambda(val, threshold))

        return reward

    def _target_shaping(self, state) -> float:
        reward = 0.0

        safety_weight = 0.0
        for spec in self._safety_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = eval_var(state, var, fn)

            safety_weight *= float(cmp_lambda(val, threshold))

        for spec in self._target_spec:
            (fn, var), aritm_op, threshold = spec._predicate
            val = eval_var(state, var, fn)

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                target_reward = clip_and_norm(val, minv, threshold)
            elif aritm_op in ["<", "<="]:
                target_reward = 1.0 - clip_and_norm(val, threshold, maxv)

            reward += safety_weight * target_reward

        return reward

    def _comfort_shaping(self, state) -> float:
        reward = 0.0

        safety_weight = 1.0
        for spec in self._safety_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = eval_var(state, var, fn)

            safety_weight *= float(cmp_lambda(val, threshold))

        target_weight = 1.0
        for spec in self._target_spec:
            (fn, var), aritm_op, threshold = spec._predicate
            val = eval_var(state, var, fn)

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                target_reward = clip_and_norm(val, minv, threshold)
            elif aritm_op in ["<", "<="]:
                target_reward = 1.0 - clip_and_norm(val, threshold, maxv)

            target_weight *= target_reward

        for spec in self._comfort_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            val = eval_var(state, var, fn)

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                comfort_reward = clip_and_norm(val, minv, threshold)
            elif aritm_op in ["<", "<="]:
                comfort_reward = 1.0 - clip_and_norm(val, threshold, maxv)

            reward += safety_weight * target_weight * comfort_reward

        return reward
