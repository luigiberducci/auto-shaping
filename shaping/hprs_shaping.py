import gymnasium

from shaping import RewardSpec
from shaping.utils.utils import clip_and_norm

_cmp_lambdas = {
    "<": lambda x, y: x < y,
    ">": lambda x, y: x > y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "==": lambda x, y: (x - y) < 1e-6,
    "!=": lambda x, y: (x - y) > 1e-6,
}


class SparseSuccessRewardWrapper(gymnasium.Wrapper):
    """
    This wrapper provides a reward of 1.0 every time the agent satisfies the target condition.
    """

    def __init__(
        self, env: gymnasium.Env, spec: RewardSpec,
    ):
        super(SparseSuccessRewardWrapper, self).__init__(env)
        self._target_specs = [
            spec for spec in spec.specs if spec._operator in ["achieve", "conquer"]
        ]
        self._variables = spec.variables

        assert (
            type(env.observation_space) == gymnasium.spaces.Dict
        ), "Observation space must be a dictionary, use DictWrapper"

    def _reward(self, state, action, next_state, done, info):
        reward = 0.0
        for target_spec in self._target_specs:
            var, aritm_op, threshold = target_spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]

            reward += float(cmp_lambda(next_state[var], threshold))
        return reward

    def step(self, action):
        next_obs, _, done, truncated, info = super().step(action)
        reward = self._reward(None, action, next_obs, done, info)
        return next_obs, reward, done, truncated, info


class HPRSWrapper(SparseSuccessRewardWrapper):
    def __init__(
        self, env: gymnasium.Env, spec: RewardSpec, gamma: float = 1.0,
    ):
        super(HPRSWrapper, self).__init__(env, spec)
        self._spec = spec
        self._gamma = gamma

        # safety, target, comfort
        self._safety_specs = [
            spec for spec in spec.specs if spec._operator in ["ensure"]
        ]
        self._target_spec = [
            spec for spec in spec.specs if spec._operator in ["achieve", "conquer"]
        ]
        self._comfort_specs = [
            spec for spec in spec.specs if spec._operator in ["encourage"]
        ]

        assert (
            len(self._target_spec) == 1
        ), f"There should be exactly one target specification, got {len(self._target_spec)}"

        self._obs = None

    def reset(self):
        self._obs, info = super().reset()
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
            var, aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]

            reward += float(cmp_lambda(state[var], threshold))

        return reward

    def _target_shaping(self, state) -> float:
        reward = 0.0

        safety_weight = 0.0
        for spec in self._safety_specs:
            var, aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]

            safety_weight *= float(cmp_lambda(state[var], threshold))

        for spec in self._target_spec:
            var, aritm_op, threshold = spec._predicate

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                target_reward = clip_and_norm(state[var], minv, threshold)
            elif aritm_op in ["<", "<="]:
                target_reward = 1.0 - clip_and_norm(state[var], threshold, maxv)

            reward += safety_weight * target_reward

        return reward

    def _comfort_shaping(self, state) -> float:
        reward = 0.0

        safety_weight = 0.0
        for spec in self._safety_specs:
            var, aritm_op, threshold = spec._predicate
            cmp_lambda = _cmp_lambdas[aritm_op]

            safety_weight *= float(cmp_lambda(state[var], threshold))

        target_weight = 0.0
        for spec in self._target_spec:
            var, aritm_op, threshold = spec._predicate

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                target_reward = clip_and_norm(state[var], minv, threshold)
            elif aritm_op in ["<", "<="]:
                target_reward = 1.0 - clip_and_norm(state[var], threshold, maxv)

            target_weight += target_reward

        for spec in self._comfort_specs:
            var, aritm_op, threshold = spec._predicate

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if aritm_op in [">", ">="]:
                comfort_reward = clip_and_norm(state[var], minv, threshold)
            elif aritm_op in ["<", "<="]:
                comfort_reward = 1.0 - clip_and_norm(state[var], threshold, maxv)

            reward += safety_weight * target_weight * comfort_reward

        return reward