from typing import Union

import gymnasium
import numpy as np

from shaping import RewardSpec
from shaping.spec.reward_spec import Variable, Constant
from shaping.utils.utils import clip_and_norm, extend_state

_cmp_lambdas = {
    "<": lambda x, y: x < y,
    ">": lambda x, y: x > y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "==": lambda x, y: (x - y) < 1e-6,
    "!=": lambda x, y: (x - y) > 1e-6,
}

fns = {
    "abs": np.abs,
    "exp": np.exp,
    None: lambda x: x,
}


class SparseSuccessRewardWrapper(gymnasium.Wrapper):
    """
    This wrapper provides a reward of 1.0 every time the agent satisfies the target condition.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[Variable],
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
        self._constants = self._spec.constants

    def _reward(self, state, action, next_state, done, info):
        reward = 0.0

        for target_spec in self._target_specs:
            (fn, var_name), aritm_op, threshold = target_spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]

            val = fns[fn](next_state[var_name])

            reward += float(cmp_lambda(val, threshold))
        return reward

    def step(self, action):
        next_obs, _, done, truncated, info = super().step(action)
        ext_state = extend_state(env=self, state=next_obs, action=action, spec=self._spec)
        reward = self._reward(None, action, ext_state, done, info)
        return next_obs, reward, done, truncated, info


class HPRSWrapper(SparseSuccessRewardWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        specs: list[str],
        variables: list[Variable],
        constants: list[Constant] = None,
        gamma: float = 1.0,
    ):
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, specs=specs, variables=variables, constants=constants, gamma=gamma,
        )
        super(HPRSWrapper, self).__init__(env, specs, variables, constants)

        self._gamma = gamma

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
        obs, info = super().reset(**kwargs)
        action0 = np.zeros_like(self.action_space.sample())
        self._obs = extend_state(env=self, state=obs, action=action0, spec=self._spec)
        return obs, info

    def step(self, action):
        next_obs, _, done, truncated, info = super().step(action)
        ext_next_obs = extend_state(env=self, state=next_obs, action=action, spec=self._spec)
        reward = self._hprs_reward(self._obs, action, ext_next_obs, done, info)
        self._obs = ext_next_obs
        return next_obs, reward, done, truncated, info

    def _hprs_reward(self, state, action, next_state, done, info):
        base_reward = super()._reward(state, action, next_state, done, info)

        # return base_reward for terminal states
        if done:
            return base_reward

        # hierarchical shaping
        safety_shaping = self._gamma * self._safety_shaping(
            next_state
        ) - self._safety_shaping(state)
        target_shaping = self._gamma * self._target_shaping(
            next_state
        ) - self._target_shaping(state)
        comfort_shaping = self._gamma * self._comfort_shaping(
            next_state
        ) - self._comfort_shaping(state)

        return base_reward + safety_shaping + target_shaping + comfort_shaping

    def _safety_shaping(self, state) -> float:
        reward = 0.0

        for spec in self._safety_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = fns[fn](state[var])

            reward += float(cmp_lambda(val, threshold))

        return reward

    def _target_shaping(self, state) -> float:
        reward = 0.0

        safety_weight = 1.0
        for spec in self._safety_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]
            val = fns[fn](state[var])

            safety_weight *= float(cmp_lambda(val, threshold))

        for spec in self._target_spec:
            (fn, var), aritm_op, threshold = spec._predicate
            val = fns[fn](state[var])

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
            val = fns[fn](state[var])

            safety_weight *= float(cmp_lambda(val, threshold))

        target_weight = 1.0
        for spec in self._target_spec:
            (fn, var), aritm_op, threshold = spec._predicate
            val = fns[fn](state[var])

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                target_reward = clip_and_norm(val, minv, threshold)
            elif aritm_op in ["<", "<="]:
                target_reward = 1.0 - clip_and_norm(val, threshold, maxv)

            target_weight *= target_reward

        for spec in self._comfort_specs:
            (fn, var), aritm_op, threshold = spec._predicate
            val = fns[fn](state[var])

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                comfort_reward = clip_and_norm(val, minv, threshold)
            elif aritm_op in ["<", "<="]:
                comfort_reward = 1.0 - clip_and_norm(val, threshold, maxv)

            reward += safety_weight * target_weight * comfort_reward

        return reward
