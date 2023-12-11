import unittest

from auto_shaping.bhnr_shaping import BHNRWrapper
from tests.utility_functions import (
    get_cartpole_example2_spec,
    get_cartpole_example1_spec,
)


class TestBHNR(unittest.TestCase):
    def test_cartpole_simple_spec(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example1_spec()
        env = BHNRWrapper(env, specs=specs, constants=constants, variables=variables)

        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())

            self.assertTrue(
                reward >= 0.0, f" negative reward of {reward}, expected >= 0.0"
            )
            self.assertTrue(
                reward <= 1.0, f" too large reward of {reward}, expected <= 1.0"
            )

        env.close()

    def test_cartpole_complex_spec(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example2_spec()
        env = BHNRWrapper(env, specs=specs, variables=variables, constants=constants)

        obs, info = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())

            self.assertTrue(
                reward >= 0.0, f" negative reward of {reward}, expected >= 0.0"
            )
            self.assertTrue(
                reward <= 1.0, f" too large reward of {reward}, expected <= 1.0"
            )

        env.close()
