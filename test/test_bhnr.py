import unittest

from shaping.bhnr_shaping import BHNRWrapper


class TestBHNR(unittest.TestCase):
    def test_cartpole_x(self):
        import gymnasium
        from shaping.utils.dictionary_wrapper import DictWrapper

        env = gymnasium.make("CartPole-v1", render_mode="human")
        env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

        specs = [
            'ensure abs "x" <= 2.4',
            #'achieve abs "x" <= 0.05',
            #'encourage abs "theta" <= 0.0',
        ]
        variables = [
            ("x", -2.4, 2.4),
            ("x_dot", -3.0, 3.0),
            ("theta", -0.2, 0.2),
            ("theta_dot", -3.0, 3.0),
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
