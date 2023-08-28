import numpy as np
from lark import Transformer


class RewardShapingTransformer(Transformer):
    list = list
    assignments = {}

    def add_constant(self, name, value):
        self.assignments[name] = value

    def assignment(self, s):
        var, val = s
        var = var.replace('"', "")
        self.add_constant(var, float(val))
        return (var, float(val))

    def gt(self, s):
        var, val = s
        var = var.replace('"', "")

        return (var, ">", float(val))

    def lt(self, s):
        var, val = s
        var = var.replace('"', "")
        return (var, "<", float(val))

    def ge(self, s):
        var, val = s
        var = var.replace('"', "")
        return (var, ">=", float(val))

    def le(self, s):
        var, val = s
        var = var.replace('"', "")
        return (var, "<=", float(val))

    def number(self, n):
        (n,) = n
        return float(n)

    def pos(self, n):
        (n,) = n
        return float(n)

    def neg(self, n):
        (n,) = n
        return -float(n)

    def string(self, s):
        var = s[0].value.replace('"', "")

        if not var in self.assignments:
            raise ValueError(f"Variable {var} not defined")

        return float(self.assignments[var])
