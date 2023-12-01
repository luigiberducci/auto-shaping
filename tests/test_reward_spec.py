import unittest

from auto_shaping.spec.reward_spec import Variable, Constant


class TestRewardSpec(unittest.TestCase):
    def test_constants(self):
        import gymnasium
        from auto_shaping import RewardSpec

        env = gymnasium.make("CartPole-v1")

        with self.assertRaises(ValueError):
            spec = RewardSpec(
                specs=['ensure "x" < "x_limit"'],
                variables=[Variable(name="x", fn="state[0]", min=-2.4, max=2.4)],
                constants=None,
            )

        # test with constant defined
        spec = RewardSpec(
            specs=['ensure "x" < "x_limit"'],
            variables=[Variable(name="x", fn="state[0]", min=-2.4, max=2.4)],
            constants=[Constant(name="x_limit", value=2.4)],
        )

        self.assertTrue(True, "RewardSpec with constants failed to initialize")

        # test with constant defined
        spec = RewardSpec(
            specs=['ensure "x" > -"x_limit"'],
            variables=[Variable(name="x", fn="state[0]", min=-2.4, max=2.4)],
            constants=[Constant(name="x_limit", value=2.4)],
        )

        self.assertTrue(True, "RewardSpec with negated constant failed to initialize")
