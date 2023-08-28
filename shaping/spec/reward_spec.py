import logging
import pathlib
from collections import namedtuple
from typing import List, Tuple, Union

from lark import Lark

from shaping.parser.transformer import RewardShapingTransformer

Variable = namedtuple("Variable", ["name", "min", "max", "description"], defaults=[""])


class RequirementSpec:
    grammar_path = pathlib.Path(__file__).parent.parent / "parser" / "grammar.txt"
    transformer = RewardShapingTransformer()
    parser = None

    def __init__(self, spec: str):
        if self.parser is None:
            with open(self.grammar_path, 'r') as f:
                grammar = f.read()
                self.parser = Lark(grammar, start='start', parser='lalr', transformer=self.transformer)

            logging.debug(f"Loaded grammar from {self.grammar_path}")

        tree = self.parser.parse(spec)
        self._operator = tree.data
        self._predicate = tree.children[0]

    def __str__(self):
        return f"{self._operator} {self._predicate}"

    def to_rtamt(self):
        pred_str = "".join([str(token) for token in self._predicate])

        if self._operator == "achieve":
            return f"eventually {pred_str}"
        elif self._operator == "conquer":
            return f"eventually always {pred_str}"
        elif self._operator == "ensure":
            return f"always {pred_str}"
        elif self._operator == "encourage":
            raise NotImplementedError("Encourage not implemented")



class RewardSpec:
    def __init__(
        self,
        specs: List[str],
        variables: List[Union[Variable, tuple[str, float, float], tuple[str, float, float, str]]],
    ):
        assert len(specs) > 0, "At least one specification must be provided"
        assert len(variables) > 0, "At least one variable must be provided"

        self._specs = [RequirementSpec(sp) for sp in specs]

        self._variables = []
        for var in variables:
            if isinstance(var, Variable):
                assert var.min < var.max, f"Variable {var.name} has min value greater than max value"
            elif isinstance(var, tuple):
                assert len(var) in [3, 4], f"Variable {var[0]} must be a tuple of length 3 or 4"
                assert var[1] < var[2], f"Variable {var[0]} has min value greater than max value"
            else:
                raise ValueError(f"Variable {var} must be a Variable or a tuple")
            self._variables.append(Variable(*var))

    @property
    def specs(self) -> List[RequirementSpec]:
        return self._specs

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @staticmethod
    def from_yaml(file: pathlib.Path) -> "RewardSpec":
        import yaml

        with open(file, "r") as f:
            spec = yaml.safe_load(f)

        specs = spec["specs"]

        variables = []
        for var_dict in spec["variables"]:
            assert all([k in var_dict for k in ["name", "min", "max"]]), f"Variable {var_dict} must have keys 'name', 'min', and 'max'"
            desc = var_dict.get("description", "")
            var = Variable(name=var_dict["name"], min=var_dict["min"], max=var_dict["max"], description=desc)
            variables.append(var)

        return RewardSpec(
            specs=specs,
            variables=variables
        )

