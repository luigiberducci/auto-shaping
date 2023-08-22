from typing import List, Tuple


class RewardSpec:
    def __init__(
        self,
        specs: List[str],
        variables: List[str],
        ranges: List[Tuple[float, float]],
    ):
        assert len(specs) > 0, "At least one specification must be provided"
        assert len(variables) == len(ranges), "The number of variables and ranges must be the same"

        self._specs = specs
        self._variables = variables
        self._ranges = ranges

    @property
    def specs(self) -> List[str]:
        return self._specs

    @property
    def variables(self) -> List[str]:
        return self._variables

    @property
    def ranges(self) -> List[Tuple[float, float]]:
        return self._ranges
