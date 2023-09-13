from typing import Any, List, Dict, TypeVar

import numpy as np


def monitor_stl_episode(
    stl_spec: str, vars: List[str], episode: Dict[str, Any], types: List[str] = None
):
    import rtamt

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


KeyType = TypeVar("KeyType")


def deep_update(
    mapping: dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
