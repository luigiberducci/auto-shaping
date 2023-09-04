import unittest

from shaping.tltl_shaping import TLTLWrapper


class TestTLTL(unittest.TestCase):

    def test_cartpole_x(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode="human")

        specs=['ensure "x" < 2.4', 'ensure "x" > -2.4',]
        variables=[
            ("x", -2.4, 2.4),
            ("x_dot", -3.0, 3.0),
            ("theta", -0.2, 0.2),
            ("theta_dot", -3.0, 3.0),
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

        from shaping.spec.reward_spec import RewardSpec

        specs=[
                'ensure "x" < 2.4',
                'ensure "x" > -2.4',
                'ensure "theta" < 0.2',
                'ensure "theta" > -0.2',
            ]
        variables=[
                ("x", -2.4, 2.4),
                ("x_dot", -3.0, 3.0),
                ("theta", -0.2, 0.2),
                ("theta_dot", -3.0, 3.0),
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
