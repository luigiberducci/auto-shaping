import os
import unittest

import numpy as np

from shaping.utils import monitor_stl_episode
from shaping.utils.dictionary_wrapper import DictWrapper


class TestRTAMTUtils(unittest.TestCase):
    def test_stl_monitor(self):
        import rtamt
        import pandas as pd

        test_dir = os.path.dirname(os.path.abspath(__file__))
        data1 = pd.read_csv(f"{test_dir}/example1.csv", delimiter=",", header=0)

        # from original example
        spec = rtamt.StlDiscreteTimeSpecification(semantics=rtamt.Semantics.STANDARD)
        spec.name = "Example 1"
        spec.declare_var("req", "float")
        spec.declare_var("gnt", "float")
        spec.declare_var("out", "float")
        spec.set_var_io_type("req", "input")
        spec.set_var_io_type("gnt", "output")
        spec.spec = "out = ((req>=3) implies (eventually[0:5](gnt>=3)))"
        # try:
        spec.parse()
        spec.pastify()
        # except rtamt.RTAMTException as err:
        #    print("RTAMT Exception: {}".format(err))
        #    exit()

        for i in range(len(data1)):
            rob = spec.update(i, [("req", data1["req"][i]), ("gnt", data1["gnt"][i])])

        # from custom utils
        datadict = {k: data1[k].tolist() for k in data1.columns}
        rob_trace = monitor_stl_episode(
            stl_spec=spec.spec,
            vars=["req", "gnt"],
            types=["float", "float"],
            episode=datadict,
        )
        rob1 = rob_trace[0][1]

        # check that the robustness is the same
        assert np.isclose(
            rob, rob1, atol=1e-5
        ), f"robustness is not correct, expected {rob}, got {rob1}"

    def test_stl_monitor_notypes(self):
        import rtamt
        import pandas as pd

        test_dir = os.path.dirname(os.path.abspath(__file__))
        data1 = pd.read_csv(f"{test_dir}/example1.csv", delimiter=",", header=0)

        # from original example
        spec = rtamt.StlDiscreteTimeSpecification(semantics=rtamt.Semantics.STANDARD)
        spec.name = "Example 1"
        spec.declare_var("req", "float")
        spec.declare_var("gnt", "float")
        spec.declare_var("out", "float")
        spec.set_var_io_type("req", "input")
        spec.set_var_io_type("gnt", "output")
        spec.spec = "out = ((req>=3) implies (eventually[0:5](gnt>=3)))"
        # try:
        spec.parse()
        spec.pastify()
        # except rtamt.RTAMTException as err:
        #    print("RTAMT Exception: {}".format(err))
        #    exit()

        for i in range(len(data1)):
            rob = spec.update(i, [("req", data1["req"][i]), ("gnt", data1["gnt"][i])])

        # from custom utils
        datadict = {k: data1[k].tolist() for k in data1.columns}
        rob_trace = monitor_stl_episode(
            stl_spec=spec.spec, vars=["req", "gnt"], episode=datadict
        )
        rob1 = rob_trace[0][1]

        # check that the robustness is the same
        assert np.isclose(
            rob, rob1, atol=1e-5
        ), f"robustness is not correct, expected {rob}, got {rob1}"


class TestWrappers(unittest.TestCase):
    def test_dict_wrapper(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        self.assertTrue(
            hasattr(env, "observation_space"), "observation space not found"
        )
        self.assertTrue(
            type(env.observation_space) == gymnasium.spaces.Dict,
            "observation space not a dict",
        )

        done = False
        obs, info = env.reset()
        self.assertTrue(type(obs) == dict, "observation after reset not a dict")

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            self.assertTrue(
                type(obs) == dict, f"observation after step not a dict, got {type(obs)}"
            )

        env.close()


class TestRewardTypes(unittest.TestCase):
    def test_enum_types(self):
        from shaping import RewardType

        reward_types = ["TLTL", "BHNR", "HPRS", "PAM", "RPR"]

        for reward_type in reward_types:
            assert (
                reward_type in RewardType.__members__
            ), f"reward type {reward_type} not in RewardType enum"
            reward = RewardType.from_str(reward_type)
            assert (
                reward.name == reward_type
            ), f"reward type {reward_type} not in RewardType enum"
