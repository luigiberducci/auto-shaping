import unittest


class TestRewardSpec(unittest.TestCase):
    def test_constants(self):
        import gymnasium
        from shaping import RewardSpec

        env = gymnasium.make("CartPole-v1")

        with self.assertRaises(ValueError):
            spec = RewardSpec(
                specs=['ensure "x" < "x_limit"'],
                variables=[("x", -2.4, 2.4)],
                constants=None,
            )

        # test with constant defined
        spec = RewardSpec(
            specs=['ensure "x" < "x_limit"'],
            variables=[("x", -2.4, 2.4)],
            constants=[("x_limit", 2.4)],
        )

        self.assertTrue(True, "RewardSpec with constants failed to initialize")

        # test with constant defined
        spec = RewardSpec(
            specs=['ensure "x" > -"x_limit"'],
            variables=[("x", -2.4, 2.4)],
            constants=[("x_limit", 2.4)],
        )

        self.assertTrue(True, "RewardSpec with negated constant failed to initialize")
