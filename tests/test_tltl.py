import unittest

from shaping.spec.reward_spec import Variable
from shaping.tltl_shaping import TLTLWrapper
from shaping.utils.dictionary_wrapper import DictWrapper


class TestTLTL(unittest.TestCase):
    def test_cartpole_x(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        specs = [
            'ensure "x" < 2.4',
            'ensure "x" > -2.4',
        ]
        variables = [
            Variable(name="x", min=-2.4, max=2.4),
            Variable(name="x_dot", min=-3.0, max=3.0),
            Variable(name="theta", min=-0.2, max=0.2),
            Variable(name="theta_dot", min=-3.0, max=3.0),
        ]

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

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        specs = [
            'ensure "x" < 2.4',
            'ensure "x" > -2.4',
            'ensure "theta" < 0.2',
            'ensure "theta" > -0.2',
        ]
        variables = [
            Variable(name="x", min=-2.4, max=2.4),
            Variable(name="x_dot", min=-3.0, max=3.0),
            Variable(name="theta", min=-0.2, max=0.2),
            Variable(name="theta_dot", min=-3.0, max=3.0),
        ]

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
