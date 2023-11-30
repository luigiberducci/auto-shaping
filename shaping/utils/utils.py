from typing import Any, List, Dict

import gymnasium
import rtamt
import numpy as np

from shaping.spec.reward_spec import RewardSpec


def monitor_stl_episode(
    stl_spec: str, vars: List[str], episode: Dict[str, Any], types: List[str] = None
):

    assert type(episode) == dict, f"episode must be a dict, got {type(episode)}"

    if types is None:
        types = ["float"] * len(vars)

    spec = rtamt.STLSpecification()
    for v, t in zip(vars, types):
        spec.declare_var(v, f"{t}")

    spec.spec = stl_spec
    spec.parse()

    # sanity check: all list in the episode dict have same length
    lengths = set([len(v) for v in episode.values()])
    assert (
        len(lengths) == 1
    ), f"all lists in the episode dict must have same length, got {lengths}"
    assert lengths.pop() > 1, f"episode must have at least 2 steps, got {lengths}"
    robustness_trace = spec.evaluate(episode)

    return robustness_trace


def monitor_filtering_stl_episode(
    stl_spec: str, vars: List[str], episode: Dict[str, Any], types: List[str] = None
):
    assert type(episode) == dict, f"episode must be a dict, got {type(episode)}"

    if types is None:
        types = ["float"] * len(vars)

    spec = rtamt.FilteringStlDiscreteTimeOfflineSpecification()
    for v, t in zip(vars, types):
        spec.declare_var(v, f"{t}")

    spec.spec = stl_spec
    spec.parse()

    # sanity check: all list in the episode dict have same length
    lengths = set([len(v) for v in episode.values()])
    assert (
        len(lengths) == 1
    ), f"all lists in the episode dict must have same length, got {lengths}"
    assert lengths.pop() > 1, f"episode must have at least 2 steps, got {lengths}"
    robustness_trace = spec.evaluate(episode)

    return robustness_trace


def clip_and_norm(v: float, minv: float, maxv: float) -> float:
    """
    Normalize value in [0, 1].

    :param v: value `v` before normalization,
    :param minv: `minv` extreme values of the domain.
    :param maxv: `maxv` extreme values of the domain.
    :return: normalized value v' in [0, 1].
    """
    if np.abs(minv - maxv) <= 0.000001:
        return 0.0

    # rewritten to avoid clip
    if v < minv:
        v = minv
    elif v > maxv:
        v = maxv

    return (v - minv) / (maxv - minv)


def extend_state(env: gymnasium.Env, state: dict, action: np.ndarray, spec: RewardSpec) -> dict:
    """
    Given a state and a reward specification, return an extended state with all the constants and variables.
    """
    context = {}
    # first add the constants
    for const_name, const in spec._constants.items():
        value = const.value
        if isinstance(value, str):
            value = float(eval(value, {**context, "env": env, "np": np}))
        context[const_name] = value
    # then add the variables from the state and action
    context.update({"state": state})
    context.update({"action": action})
    # finally compute derived variables
    for var_name, var in spec._variables.items():
        value = float(eval(var.fn, {**context, "env": env, "np": np}))
        context[var_name] = value
    return context
