import logging
import pathlib
from collections import namedtuple
from typing import List, Tuple, Union

from lark import Lark

from shaping.parser.transformer import RewardShapingTransformer

Variable = namedtuple(
    "Variable", ["name", "fn", "min", "max", "description"], defaults=[None] * 5
)
Constant = namedtuple("Constant", ["name", "value", "description"], defaults=[None] * 3)


def check_var(name, min, max, fn=None, description=None):
    assert isinstance(name, str), f"Variable name must be a string, got {type(name)}"
    assert isinstance(
        min, (int, float)
    ), f"Variable min must be a number, got {type(min)}"
    assert isinstance(
        max, (int, float)
    ), f"Variable max must be a number, got {type(max)}"
    assert min < max, f"Variable has min value greater than max value"
    assert isinstance(
        fn, str
    ), f"Variable fn must be a expression as string, got {type(fn)}"
    assert description is None or isinstance(
        description, str
    ), f"Variable description must be a string or None, got {type(description)}"


def check_const(name, value, description=None):
    assert isinstance(name, str), f"Constant name must be a string, got {type(name)}"
    assert isinstance(
        value, (int, float, str)
    ), f"Constant value must be a number, got {type(value)}"
    assert description is None or isinstance(
        description, str
    ), f"Constant description must be a string or None, got {type(description)}"


class RequirementSpec:
    grammar_path = pathlib.Path(__file__).parent.parent / "parser" / "grammar.txt"
    transformer = RewardShapingTransformer()
    parser = None

    def __init__(self, spec: str):
        if self.parser is None:
            with open(self.grammar_path, "r") as f:
                grammar = f.read()
                self.parser = Lark(
                    grammar, start="start", parser="lalr", transformer=self.transformer
                )

            logging.debug(f"Loaded grammar from {self.grammar_path}")

        tree = self.parser.parse(spec)
        self._spec = spec
        self._operator = tree.data
        self._predicate = tree.children[0]

    def __str__(self):
        return self._spec

    def to_rtamt(self):
        # unpack predicate
        proc_fn, var = self._predicate[0]
        cmp_op = self._predicate[1]
        threshold_value = self._predicate[2]

        # process var
        var_str = ""
        if proc_fn == "abs":
            var_str = f"abs({var})"
        elif proc_fn == "exp":
            var_str = f"exp({var})"
        elif proc_fn is None:
            var_str = f"{var}"
        else:
            raise ValueError(f"Unknown processing function {proc_fn}")
        pred_str = f"{var_str} {cmp_op} {threshold_value}"

        # process temporal operator
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
        variables: List[Variable],
        constants: List[Constant] = None,
    ):
        assert len(specs) > 0, "At least one specification must be provided"
        assert len(variables) > 0, "At least one variable must be provided"

        self._variables = {}
        for var in variables:
            assert isinstance(
                var, Variable
            ), f"Variable {var} must be a Variable. Got {type(var)}"
            check_var(var.name, var.min, var.max, var.fn, var.description)

            self._variables[var.name] = var

        self._constants = {}
        if constants is not None:
            for const in constants:
                assert isinstance(
                    const, Constant
                ), f"Constant {const} must be a Constant. Got {type(const)}"
                check_const(const.name, const.value, const.description)

                self._constants[const.name] = const
                RequirementSpec.transformer.add_constant(
                    name=const.name, value=const.value
                )

        # important to do this after constants are set
        self._specs = [RequirementSpec(sp) for sp in specs]

    @property
    def specs(self) -> List[RequirementSpec]:
        return self._specs

    @property
    def variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def constants(self) -> dict[str, Constant]:
        return self._constants

    @staticmethod
    def from_yaml(file: pathlib.Path) -> "RewardSpec":
        import yaml

        with open(file, "r") as f:
            spec = yaml.safe_load(f)

        specs = spec["specs"]

        variables = []
        for var_dict in spec["variables"]:
            assert all(
                [k in var_dict for k in ["name", "min", "max"]]
            ), f"Variable {var_dict} must have keys 'name', 'min', and 'max'"
            desc = var_dict.get("description", None)
            fn = var_dict.get("fn", None)

            var = Variable(
                name=var_dict["name"],
                min=var_dict["min"],
                max=var_dict["max"],
                fn=fn,
                description=desc,
            )

            check_var(var.name, var.min, var.max, var.fn, var.description)
            variables.append(var)

        constants = []
        if "constants" in spec:
            for const_dict in spec["constants"]:
                assert all(
                    [k in const_dict for k in ["name", "value"]]
                ), f"Constant {const_dict} must have keys 'name' and 'value'"
                desc = const_dict.get("description", "")
                const = Constant(
                    name=const_dict["name"],
                    value=const_dict["value"],
                    description=desc,
                )

                check_const(const.name, const.value, const.description)
                constants.append(const)

        return RewardSpec(specs=specs, variables=variables, constants=constants)
