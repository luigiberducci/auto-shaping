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
        return (var, ">", float(val))

    def lt(self, s):
        var, val = s
        return (var, "<", float(val))

    def ge(self, s):
        var, val = s
        return (var, ">=", float(val))

    def le(self, s):
        var, val = s
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

    def abs(self, s):
        var = s[0].value.replace('"', "")
        return ("abs", var)

    def exp(self, s):
        var = s[0].value.replace('"', "")
        return ("exp", var)

    def var(self, s):
        var = s[0].value.replace('"', "")
        return (None, var)

    def string(self, s):
        var = s[0].value.replace('"', "")

        if not var in self.assignments:
            raise ValueError(f"Variable {var} not defined")

        if isinstance(self.assignments[var], str):
            assert False
            evaluated = eval(self.assignments[var], self.assignments)
        else:
            evaluated = self.assignments[var]
        return float(evaluated)
