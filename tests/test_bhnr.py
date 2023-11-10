import unittest

from shaping.bhnr_shaping import BHNRWrapper
from shaping.spec.reward_spec import Variable


class TestBHNR(unittest.TestCase):
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

        env = BHNRWrapper(env, specs=specs, variables=variables)

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
