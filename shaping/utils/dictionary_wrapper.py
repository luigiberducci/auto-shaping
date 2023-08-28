from typing import Callable, Any

import gymnasium
from gymnasium.core import ObsType, WrapperObsType


class DictWrapper(gymnasium.Wrapper):
    """
    Converts a vector observation into a dictionary observation.
    """

    def __init__(
        self, env: gymnasium.Env, variables: list[str], extractor_fn: Callable = None,
    ):
        super(DictWrapper, self).__init__(env)

        if type(self.observation_space) != gymnasium.spaces.Box:
            raise ValueError("Observation space must be a Box")
        if len(self.observation_space.shape) != 1:
            raise ValueError(
                f"Observation space must be a vector, got {self.observation_space.shape}"
            )

        obsspace = {}
        for i, var in enumerate(variables):
            low = (
                self.observation_space.low
                if len(self.observation_space.low.shape) == 1
                else self.observation_space.low[i]
            )
            high = (
                self.observation_space.high
                if len(self.observation_space.high.shape) == 1
                else self.observation_space.high[i]
            )
            obsspace[var] = gymnasium.spaces.Box(
                low=low, high=high, dtype=self.observation_space.dtype,
            )

        self.observation_space = gymnasium.spaces.Dict(obsspace)

        if extractor_fn is None:
            extractor_fn = lambda obs, done, info: {
                v: obs[i] for i, v in enumerate(variables)
            }
        self._variables = variables
        self._extractor_fn = extractor_fn

    def step(
        self, action
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = super().step(action)

        obs = self._extractor_fn(obs, done, info)

        return obs, reward, done, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(**kwargs)

        obs = self._extractor_fn(obs, False, info)

        return obs, info
