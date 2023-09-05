from collections import deque
from typing import List, Callable

import gymnasium

from shaping.utils.dictionary_wrapper import DictWrapper


class CollectionWrapper(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """
    Collects k-th most recent observable varibles over an episode.

    :param env: the environment to wrap
    :param variables: the list of observable variables to collect from the state
    :param extractor_fn: a function that extracts the variables from the state
    :param window_len: the length k of the window to collect the variables, if None, the whole episode is collected
    """

    def __init__(
        self,
        env: gymnasium.Env,
        variables: List[str],
        extractor_fn: Callable = None,
        window_len: int = None,
    ):
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, variables=variables, extractor_fn=extractor_fn, window_len=window_len,
        )

        if isinstance(env.observation_space, gymnasium.spaces.Box):
            self._env = DictWrapper(env, variables=variables, extractor_fn=extractor_fn)

        super(CollectionWrapper, self).__init__(env)

        self._variables = variables
        self._extractor_fn = extractor_fn
        self._window_len = window_len
        self._flag_time = "time" in self._variables
        self._time = None
        self._episode = None

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)

        self._episode = {var: deque(maxlen=self._window_len) for var in self._variables}
        if not self._flag_time:
            self._episode["time"] = deque(maxlen=self._window_len)
            self._time = 0.0

        return state, info

    def step(self, action):
        if self._episode is None:
            raise RuntimeError("reset() must be called before step()")

        obs, reward, done, truncated, info = super().step(action)

        # collect observable variables from the state
        for key, value in obs.items():
            self._episode[key].append(value)

        if not self._flag_time:
            self._time += 1.0
            self._episode["time"].append(self._time)

        return obs, reward, done, truncated, info
