import unittest

import numpy as np

from shaping.hprs_shaping import HPRSWrapper
from shaping.spec.reward_spec import Variable


class TestHPRS(unittest.TestCase):
    def test_cartpole_x(self):
        import gymnasium
        from shaping.utils.dictionary_wrapper import DictWrapper

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        specs = [
            'ensure abs "x" <= 2.4',
            'achieve abs "x" <= 0.05',
            'encourage abs "theta" <= 0.0',
        ]
        variables = [
            Variable(name="x", min=-2.4, max=2.4),
            Variable(name="x_dot", min=-3.0, max=3.0),
            Variable(name="theta", min=-0.2, max=0.2),
            Variable(name="theta_dot", min=-3.0, max=3.0),
        ]

        env = HPRSWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())

        env.close()
