import unittest

import numpy as np

from shaping.hprs_shaping import HPRSWrapper


class TestHPRS(unittest.TestCase):
    def test_cartpole_x(self):
        import gymnasium
        from shaping.utils.dictionary_wrapper import DictWrapper

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        from shaping.spec.reward_spec import RewardSpec

        spec = RewardSpec(
            specs=['ensure abs "x" <= 2.4',
                   'achieve abs "x" <= 0.05',
                   'encourage abs "theta" <= 0.0'],
            variables=[
                ("x", -2.4, 2.4),
                ("x_dot", -3.0, 3.0),
                ("theta", -0.2, 0.2),
                ("theta_dot", -3.0, 3.0),
            ],
        )
        env = HPRSWrapper(env, spec)

        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            print(reward)

        env.close()

