import numpy as np
from lark import Transformer


class RewardShapingTransformer(Transformer):
    list = list
    assignments = {}

    def assignment(self, s):
        var, val = s
        var = var.replace('"', "")
        self.assignments[var] = float(val)
        return (var, float(val))

    def gt(self, s):
        var, val = s
        var = var.replace('"', "")

        if not isinstance(val, float):
            val = val.children[0].value.replace('"', "")
            assert val in self.assignments, f"Variable {val} not defined"
            val = self.assignments[val]

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

    def eq(self, s):
        var, val = s
        var = var.replace('"', "")
        return (var, "==", float(val))

    def ne(self, s):
        var, val = s
        var = var.replace('"', "")
        return (var, "!=", float(val))

    def number(self, n):
        (n,) = n
        return float(n)
