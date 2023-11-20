import pathlib
from enum import Enum
from typing import Union

import gymnasium

from shaping.bhnr_shaping import BHNRWrapper
from shaping.spec.reward_spec import RewardSpec, Variable, Constant
from shaping.tltl_shaping import TLTLWrapper
from shaping.hprs_shaping import HPRSWrapper

__entry_points__ = {
    "TLTL": TLTLWrapper,
    "HPRS": HPRSWrapper,
    "BHNR": BHNRWrapper,
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
    env_kwargs: dict = None,
):
    """
    Wrap an environment with a reward shaping wrapper.

    :param env: the environment to wrap or its name if it is a registered environment
    :param reward: the reward shaping type
    :param spec: the reward specification or the path to the yaml file, or None if registered environment with configs
    """
    if spec is None:
        assert isinstance(env, str), "spec must be provided if env is not a string"
        spec = pathlib.Path(__file__).parent.parent / "configs" / f"{env}.yaml"
        assert spec.exists(), f"spec file does not exist in configs directory, {spec}"

    if isinstance(env, str):
        env_kwargs = env_kwargs or {}
        env = gymnasium.make(env, **env_kwargs)

    if isinstance(spec, pathlib.Path) or isinstance(spec, str):
        spec = pathlib.Path(spec)
        spec = RewardSpec.from_yaml(spec)

    assert isinstance(
        spec, RewardSpec
    ), "spec must be a RewardSpec or a path to a yaml file"

    specs_str = [str(sp) for sp in spec.specs]

    if reward == "default":
        return env

    return __entry_points__[reward](
        env,
        specs=specs_str,
        variables=list(spec.variables.values()),
        constants=list(spec.constants.values()),
    )
