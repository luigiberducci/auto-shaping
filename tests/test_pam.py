import unittest

from auto_shaping import Variable, PAMWrapper
from tests.utility_functions import (
    get_cartpole_spec_within_xlim_and_balance,
    get_cartpole_example2_spec,
)


class TestPAM(unittest.TestCase):
    def test_cartpole_unsafe(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)

        specs = [
            'ensure abs "theta" < 0.2',
        ]
        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        env = PAMWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset(seed=42)
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        # expected unsafe (0.0/1.0) + missing target (0.5/0.5) + missing confort (0.25/0.25) = 0.75
        self.assertTrue(tot_reward == 0.75, f"Expected reward 0.75, got {tot_reward}")

    def test_cartpole_safe_no_target(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)

        specs = [
            'ensure abs "x" < 2.4',
            'conquer abs "x" < 0.0',    # unsatisfiable
        ]
        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        env = PAMWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset(seed=42)
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        # expected safe (1.0/1.0) + no target (0.0/0.5) + missing confort (0.25/0.25) = 1.25
        self.assertTrue(1.25 == tot_reward, f"Expected reward 1.25, got {tot_reward}")

    def test_cartpole_safe_target_no_comfort(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)

        specs = [
            'ensure abs "x" < 2.4',
            'conquer abs "x" < 0.5',    # large target area
            'encourage abs "theta" < 0.0', # impossible to sat
        ]
        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        env = PAMWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset(seed=42)
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        # expected safe (1.0/1.0) + no target (0.5/0.5) + missing confort (0.0/0.25) = 1.5
        self.assertTrue(1.5 == tot_reward, f"Expected reward 1.5, got {tot_reward}")

    def test_cartpole_safe_target_some_comfort(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)

        specs = [
            'ensure abs "x" < 2.4',
            'conquer abs "x" < 0.5',    # large target area
            'encourage abs "theta" < 0.1', # partially satisfiable from rnd agent
        ]
        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        env = PAMWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset(seed=42)
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        # expected safe (1.0/1.0) + no target (0.5/0.5) + missing confort (0.xx/0.25) = 1.5 + 0.xx
        self.assertTrue(1.5 < tot_reward, f"Expected reward >1.5, got {tot_reward}")

    def test_cartpole_safe_target_full_comfort(self):
        import gymnasium

        env = gymnasium.make("CartPole-v1", render_mode=None)

        specs = [
            'ensure abs "x" < 2.4',
            'conquer abs "x" < 0.5',    # large target area
            'encourage abs "theta" <= 0.25', # always satisfiable from rnd agent
        ]
        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        env = PAMWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset(seed=42)
        done = False
        tot_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            env.render()
            tot_reward += reward
        env.close()

        # expected safe (1.0/1.0) + no target (0.5/0.5) + missing confort (0.xx/0.25) = 1.5 + 0.xx
        self.assertTrue(1.75 <= tot_reward, f"Expected reward 1.75, got {tot_reward}")


