import pathlib
from enum import Enum
from typing import Union

import gymnasium

from shaping.spec.reward_spec import RewardSpec
from shaping.tltl_shaping import TLTLWrapper
from shaping.hprs_shaping import HPRSWrapper

__entry_points__ = {
    "TLTL": TLTLWrapper,
    "HPRS": HPRSWrapper,
}



class RewardType(Enum):
    TLTL: int = 0
    BHNR: int = 1
    HPRS: int = 2
    PAM: int = 3
    RPR: int = 4

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_str(reward_name: str) -> "RewardType":
        return RewardType[reward_name]


def wrap(
    env: Union[str, gymnasium.Env],
    reward: str,
    spec: Union[RewardSpec, str, pathlib.Path] = None,
):
    """
    Wrap an environment with a reward shaping wrapper.

    :param env: the environment to wrap or its name if it is a registered environment
    :param reward: the reward shaping type
    :param spec: the reward specification or the path to the yaml file
    """

    if isinstance(env, str):
        env = gymnasium.make(env)

    if isinstance(spec, pathlib.Path) or isinstance(spec, str):
        spec = pathlib.Path(spec)
        spec = RewardSpec.from_yaml(spec)

    assert isinstance(spec, RewardSpec), "spec must be a RewardSpec or a path to a yaml file"

    var_tuples = [(var.name, var.min, var.max) for var_name, var in spec.variables.items()]
    const_tuples = [(const.name, const.value) for const_name, const in spec.constants.items()]
    specs_str = [str(sp) for sp in spec.specs]

    return __entry_points__[reward](env, specs=specs_str, variables=var_tuples, constants=const_tuples)
