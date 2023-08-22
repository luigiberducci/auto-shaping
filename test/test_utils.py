import unittest

import numpy as np

from shaping.utils import monitor_stl_episode


class TestRTAMTUtils(unittest.TestCase):
    def test_stl_monitor(self):
        import rtamt
        import pandas as pd

        data1 = pd.read_csv("example1.csv", delimiter=",", header=0)

        # from original example
        spec = rtamt.StlDiscreteTimeSpecification(semantics=rtamt.Semantics.STANDARD)
        spec.name = "Example 1"
        spec.declare_var("req", "float")
        spec.declare_var("gnt", "float")
        spec.declare_var("out", "float")
        spec.set_var_io_type("req", "input")
        spec.set_var_io_type("gnt", "output")
        spec.spec = "out = ((req>=3) implies (eventually[0:5](gnt>=3)))"
        try:
            spec.parse()
            spec.pastify()
        except rtamt.RTAMTException as err:
            print("RTAMT Exception: {}".format(err))
            exit()

        for i in range(len(data1)):
            rob = spec.update(i, [("req", data1["req"][i]), ("gnt", data1["gnt"][i])])

        # from custom utils
        rob_trace = monitor_stl_episode(
            stl_spec=spec.spec,
            vars=["req", "gnt"],
            types=["float", "float"],
            episode=data1,
        )
        rob1 = rob_trace[0][1]

        # check that the robustness is the same
        assert np.isclose(
            rob, rob1, atol=1e-5
        ), f"robustness is not correct, expected {rob}, got {rob1}"

    def test_stl_monitor_notypes(self):
        import rtamt
        import pandas as pd

        data1 = pd.read_csv("example1.csv", delimiter=",", header=0)

        # from original example
        spec = rtamt.StlDiscreteTimeSpecification(semantics=rtamt.Semantics.STANDARD)
        spec.name = "Example 1"
        spec.declare_var("req", "float")
        spec.declare_var("gnt", "float")
        spec.declare_var("out", "float")
        spec.set_var_io_type("req", "input")
        spec.set_var_io_type("gnt", "output")
        spec.spec = "out = ((req>=3) implies (eventually[0:5](gnt>=3)))"
        try:
            spec.parse()
            spec.pastify()
        except rtamt.RTAMTException as err:
            print("RTAMT Exception: {}".format(err))
            exit()

        for i in range(len(data1)):
            rob = spec.update(i, [("req", data1["req"][i]), ("gnt", data1["gnt"][i])])

        # from custom utils
        rob_trace = monitor_stl_episode(
            stl_spec=spec.spec, vars=["req", "gnt"], episode=data1
        )
        rob1 = rob_trace[0][1]

        # check that the robustness is the same
        assert np.isclose(
            rob, rob1, atol=1e-5
        ), f"robustness is not correct, expected {rob}, got {rob1}"
