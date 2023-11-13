import unittest

from shaping.tltl_shaping import TLTLWrapper
from tests.utility_functions import get_cartpole_spec_within_xlim, get_cartpole_example1_spec, \
    get_cartpole_spec_within_xlim_and_balance, get_cartpole_example2_spec


class TestTLTL(unittest.TestCase):
    def test_cartpole_within_xlim(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_spec_within_xlim()
        env = TLTLWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        self.assertGreaterEqual(
            tot_reward,
            0.0,
            "Expected reward for staying within world limits to be non-negative",
        )

    def test_cartpole_x_theta(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_spec_within_xlim_and_balance()
        env = TLTLWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        self.assertLess(
            tot_reward, 0.0, "Expected negative reward for balancing the pole"
        )

    def test_more_complex_spec(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example2_spec()
        env = TLTLWrapper(env, specs=specs, variables=variables, constants=constants)

        obs, info = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            tot_reward += reward
        env.close()

        self.assertLess(
            tot_reward, 0.0, "Expected negative reward for balancing the pole"
        )
