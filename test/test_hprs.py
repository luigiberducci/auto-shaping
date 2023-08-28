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
            specs=['ensure "x" < 2.4', 'ensure "x" > -2.4',
                   'achieve "x" <= 0.1',
                   'encourage "theta" < 0.2', 'encourage "theta" > -0.2'],
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

            goal_reached = obs["x"] <= 0.1
            self.assertTrue(not goal_reached or np.isclose(reward, 1.0), f"Expected reward to be 1.0 when goal reached, got {reward} for x={obs['x']}")
            self.assertTrue(goal_reached or np.isclose(reward, 0.0), f"Expected reward to be 0.0 when goal not reached, got {reward} for x={obs['x']}")

        env.close()

