from typing import Any, List, Dict


def monitor_stl_episode(stl_spec: str, vars: List[str], episode: Dict[str, Any], types: List[str] = None):
    import rtamt

    assert type(episode) == dict, f"episode must be a dict, got {type(episode)}"

    if types is None:
        types = ['float'] * len(vars)

    spec = rtamt.STLSpecification()
    for v, t in zip(vars, types):
        spec.declare_var(v, f'{t}')

    spec.spec = stl_spec
    spec.parse()

    # sanity check: all list in the episode dict have same length
    lengths = set([len(v) for v in episode.values()])
    assert len(lengths) == 1, f"all lists in the episode dict must have same length, got {lengths}"
    assert lengths.pop() > 1, f"episode must have at least 2 steps, got {lengths}"
    robustness_trace = spec.evaluate(episode)

    return robustness_trace

