import random
import unittest

import numpy as np

from auto_shaping import Variable, PAMWrapper, RPRWrapper
from tests.utility_functions import (
    get_cartpole_spec_within_xlim_and_balance,
    get_cartpole_example2_spec,
)


RENDER_MODE = None
class TestRPR(unittest.TestCase):


    def test_cartpole_all_combos(self):
        """
        These test check value of RPR for 3 classes of requirements,
        for all possible combinations of S/T/C requirements sat/unsat.
        """
        import gymnasium

        seed = 42
        np.random.seed(seed)
        random.seed(seed)

        # simple specs trivially satisfied or unsatisfied in the cartpole env (note: |x| < 2.5)
        cartpole_specs = {
            "safety": {
                True: 'ensure abs "theta" < 2.5',
                False: 'ensure abs "theta" < 0.0',
            },
            "target": {
                True: 'conquer abs "x" < 2.5',
                False: 'conquer abs "x" < 0.0',
            },
            "comfort": {
                True: 'encourage abs "x" < 2.5',
                False: 'encourage abs "x" < 0.0',
            },
        }

        variables = [
            Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
            Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
            Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
            Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        ]

        all_combinations = [
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [False, True, True],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ]

        for combo in all_combinations:
            specs = [
                cartpole_specs["safety"][combo[0]],
                cartpole_specs["target"][combo[1]],
                cartpole_specs["comfort"][combo[2]],
            ]

            env = gymnasium.make("CartPole-v1", render_mode=RENDER_MODE)
            env = RPRWrapper(env, specs=specs, variables=variables)
            env.action_space.seed(0)

            for i in range(100):
                obs, info = env.reset(seed=seed)

                done = False
                tot_reward = 0.0
                while not done:
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)
                    env.render()
                    tot_reward += reward

                lb = 2.01**3 * combo[0] + 2.01**2 * combo[1] + 2.01**1 * combo[2]
                ub = lb + 1.0
                self.assertTrue(lb <= tot_reward <= ub, f"Combo: {combo}, Ep: {i}, Expected reward {lb} <= {tot_reward} <= {ub}")

            env.close()


