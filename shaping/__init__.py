import pathlib
from enum import Enum
from typing import Union

import gymnasium

from shaping.spec.reward_spec import RewardSpec


class RewardType(Enum):
    TLTL: int = 0
    BHNR: int = 1
    MORL: int = 2
    HPRS: int = 3
    PAM: int = 4
    RPR: int = 5

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_str(reward_name: str) -> "RewardType":
        return RewardType[reward_name]


def wrap(
    env: Union[str, gymnasium.Env],
    reward: str,
    spec: Union[RewardSpec, pathlib.Path] = None,
):
    """
    Wrap an environment with a reward shaping wrapper.

    :param env: the environment to wrap or its name if it is a registered environment
    :param reward: the reward shaping type
    :param spec: the reward specification or the path to the yaml file
    """

    if isinstance(env, str):
        env = gymnasium.make(env)

    if isinstance(spec, pathlib.Path):
        spec = RewardSpec.from_yaml(spec)

    if isinstance(spec, RewardSpec):
        return RewardType.from_str(reward).value(env, spec)
    else:
        raise ValueError("spec must be a RewardSpec or a path to a yaml file")
