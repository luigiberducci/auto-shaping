import random
import unittest

import gymnasium
import numpy as np

from auto_shaping.hprs_shaping import HPRSWrapper
from auto_shaping.utils.utils import extend_state
from tests.utility_functions import (
    get_cartpole_example1_spec,
    get_cartpole_example2_spec,
    get_cartpole_spec_within_xlim,
    get_bipedal_walker_safety,
    get_bipedal_walker_safety_minlidar,
    get_bipedal_walker_achieve_norm,
    get_bipedal_walker_achieve_unnorm,
    get_bipedal_walker_comfort_speed,
    get_lander_safety_only,
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
        action = np.zeros_like(env.action_space.sample())

        while not done:
            ext_obs = extend_state(env=env, state=obs, action=action, spec=env._spec)
            self.assertTrue(ext_obs == env._obs)
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

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

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = 0
            obs, reward, done, truncated, info = env.step(action)
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

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = 0
            obs, r, done, truncated, info = env.step(action)
            self.assertTrue(
                r > 0.95 or r < -0.95,
                f"expected high reward (balance) or negative penalty (fall), got {r}",
            )

        env.close()

    def test_bipedal_walker_safety(self):
        """
        Test hprs in more complex spec for cartpole env.
        """
        env = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_safety()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            obs, r, done, truncated, info = env.step(np.zeros(4))
            self.assertTrue(
                abs(r) <= 1e-6, "expected zero reward (safety on game-over)"
            )

        env.close()

    def test_bipedal_walker_safety_minlidar(self):
        """
        Test hprs in more complex spec for cartpole env.
        """
        env = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_safety_minlidar()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        tot_reward = 0.0
        while not done:
            obs, r, done, truncated, info = env.step(np.zeros(4))
            tot_reward += r
            self.assertTrue(
                r <= 0.0,
                f"expected reward to be 0.0 or -1.0 (safety on min-lidar<threshold), got {r}",
            )

        self.assertTrue(
            -1.0 <= tot_reward <= 0.0,
            f"expected tot reward to be in [-1.0, 0.0], got{tot_reward}",
        )

        env.close()

    def test_bipedal_walker_achieve_normalization(self):
        """
        Test hprs automatically normalize the rewards in the range [0, 1].
        """
        env1 = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_achieve_norm()
        env1 = HPRSWrapper(env1, specs=specs, constants=constants, variables=variables)

        env2 = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_achieve_unnorm()
        env2 = HPRSWrapper(env2, specs=specs, constants=constants, variables=variables)

        seed = 0
        obs, info = env1.reset(seed=seed)
        obs2, info = env2.reset(seed=seed)
        done = False

        while not done:
            action = np.zeros(4)
            obs, r1, done, truncated, info = env1.step(action)
            obs, r2, done, truncated, info = env2.step(action)

            self.assertTrue(
                abs(r1 - r2) < 1e-6, f"expected same reward, got err>1e-6: {abs(r1-r2)}"
            )

        env1.close()
        env2.close()

    def test_bipedal_walker_achieve_unnorm(self):
        """
        Test hprs in more complex spec for cartpole env.
        """
        env = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_achieve_unnorm()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        tot_r = 0.0
        while not done:
            obs, r, done, truncated, info = env.step(np.zeros(4))
            tot_r += r

        print("tot r:", tot_r)
        self.assertTrue(
            tot_r >= 0.0,
            f"expected bipedal walker to fall ahead, wt tot reward slightly > 0.0, got {tot_r}",
        )

        env.close()

    def test_bipedal_walker_comfort_speed(self):
        """
        Test hprs in more complex spec for cartpole env.
        """
        env = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs, constants, variables = get_bipedal_walker_comfort_speed()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        env2 = gymnasium.make("BipedalWalker-v3", render_mode=None)
        specs2, constants2, variables2 = get_bipedal_walker_achieve_unnorm()
        env2 = HPRSWrapper(
            env2, specs=specs2, constants=constants2, variables=variables2
        )

        seed = 0
        obs, info = env.reset(seed=seed)
        obs2, info = env2.reset(seed=seed)
        done = False
        done2 = False

        tot_r = 0.0
        tot_r2 = 0.0
        while not (done or done2):
            obs, r, done, truncated, info = env.step(np.zeros(4))
            obs2, r2, done2, truncated2, info2 = env2.step(np.zeros(4))
            # print(r, r2)
            tot_r += r
            tot_r2 += r2

        print("tot r:", tot_r, "tot r2:", tot_r2)
        self.assertTrue(
            tot_r >= tot_r2,
            f"expected reward to be higher with comfort speed, got {tot_r} < {tot_r2}",
        )

        env.close()

    def test_lunar_lander_safety(self):
        env = gymnasium.make("LunarLanderContinuous-v2", render_mode=None)
        specs, constants, variables = get_lander_safety_only()
        env = HPRSWrapper(env, specs=specs, constants=constants, variables=variables)

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            obs, r, done, truncated, info = env.step(np.zeros(4))
            self.assertTrue(
                abs(r) <= 1e-6, "expected zero reward (safety on game-over)"
            )

        env.close()

    def _test_lunar_lander_heuristic(self):
        import auto_shaping
        from gymnasium.envs.box2d.lunar_lander import heuristic

        env = auto_shaping.wrap(env="LunarLanderContinuous-v2", reward="HPRS")

        seed = 0
        obs, info = env.reset(seed=seed)
        done = False

        tot_r = 0.0
        while not done:
            action = heuristic(env, s=obs)
            obs, r, done, truncated, info = env.step(action)
            tot_r += r

        self.assertTrue(tot_r > 45.0, "expected tot hprs reward of heuristic agent to be > 45.0")

        env.close()
