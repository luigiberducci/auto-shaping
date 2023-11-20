import unittest

import gymnasium
import numpy as np

from shaping.hprs_shaping import HPRSWrapper
from shaping.utils.utils import extend_state
from tests.utility_functions import (
    get_cartpole_example1_spec,
    get_cartpole_example2_spec,
    get_cartpole_spec_within_xlim,
)


class TestHPRS(unittest.TestCase):
    def test_wrapper_obs(self):
        """
        Test if the wrapper correctly store the last observation to compute the potential rewards.
        """
        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example1_spec()
        env = HPRSWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset()
        done = False

        while not done:
            ext_obs = extend_state(env=env, state=obs, spec=env._spec)
            self.assertTrue(ext_obs == env._obs)
            obs, reward, done, truncated, info = env.step(env.action_space.sample())

        env.close()

    def test_cartpole_missing_achieve(self):
        """
        Check correct handling of missing achieve in the spec.
        """

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_spec_within_xlim()
        with self.assertRaises(AssertionError):
            env = HPRSWrapper(env, specs=specs, variables=variables)

    def test_cartpole_simple_spec(self):
        """
        Test hprs in simple spec for cartpole env.
        """

        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example1_spec()
        env = HPRSWrapper(env, specs=specs, variables=variables)

        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            self.assertTrue(
                reward > 0.95,
                "expected cartpole within the goal region, then reward ~ 1.0",
            )

        env.close()

    def test_cartpole_complex_spec(self):
        """
        Test hprs in more complex spec for cartpole env.
        """
        env = gymnasium.make("CartPole-v1", render_mode=None)
        specs, constants, variables = get_cartpole_example2_spec()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        obs, info = env.reset(seed=0)
        done = False

        while not done:
            obs, r, done, truncated, info = env.step(env.action_space.sample())
            self.assertTrue(
                r > 0.95 or abs(r) < 0.05,
                f"expected high reward (balance) or small negative (if falling)",
            )

        env.close()
