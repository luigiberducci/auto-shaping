import os
import pathlib
import unittest
from argparse import Namespace
from shutil import rmtree

import run


class TestRunScript(unittest.TestCase):
    _cpole_hparams_file = "../examples/hparams/ppo-CartPole-v1.yaml"
    _bw_hparams_file = "../examples/hparams/sac-BipedalWalker-v3.yaml"
    _ll_hparams_file = "../examples/hparams/sac-LunarLanderContinuous-v2.yaml"

    def _quick_test(self, **kwargs):
        if pathlib.Path("tmp-logs").exists():
            rmtree("tmp-logs")

        args_dict = {
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "train_reward": "default",
            "eval_reward": "default",
            "total_timesteps": 1e2,
            "eval_frequency": 1e5,  # no eval because total_timesteps is set to small
            "spec_file": None,
            "hparams_file": None,
            "log_dir": "tmp-logs",
            "wandb": False,
            "wandb_entity": None,
            "wandb_project": None,
            "wandb_group": None,
            "seed": 0,
        }
        args_dict.update(kwargs)
        args = Namespace(**args_dict)

        run.main(args)

        self.assertTrue(
            pathlib.Path("tmp-logs").exists(), "tmp-logs directory does not exist."
        )

        subdirs = pathlib.Path("tmp-logs").glob(f"*{args.train_reward}*")
        self.assertTrue(
            len(list(subdirs)) == 1, f"exp log dir not found among {subdirs}"
        )

        # delete tmp-logs and subdirs
        rmtree("tmp-logs")

    def test_run_ppo_cartpole_default(self):
        self._quick_test(env_id="CartPole-v1", algo="ppo", train_reward="default")

    def test_run_ppo_cartpole_HPRS(self):
        self._quick_test(env_id="CartPole-v1", algo="ppo", train_reward="HPRS")

    def test_run_ppo_cartpole_TLTL(self):
        self._quick_test(env_id="CartPole-v1", algo="ppo", train_reward="TLTL")

    def test_run_ppo_cartpole_BHNR(self):
        self._quick_test(env_id="CartPole-v1", algo="ppo", train_reward="BHNR")

    def test_run_ppo_cartpole_hparams_default(self):
        hparams_file = self._cpole_hparams_file
        self._quick_test(
            env_id="CartPole-v1",
            algo="ppo",
            train_reward="default",
            hparams_file=hparams_file,
        )

    def test_run_sac_bipedalwalker_default(self):
        self._quick_test(env_id="BipedalWalker-v3", algo="sac", train_reward="default")

    def test_run_sac_bipedalwalker_HPRS(self):
        self._quick_test(env_id="BipedalWalker-v3", algo="sac", train_reward="HPRS")

    def test_run_sac_bipedalwalker_TLTL(self):
        self._quick_test(env_id="BipedalWalker-v3", algo="sac", train_reward="TLTL")

    def test_run_sac_bipedalwalker_BHNR(self):
        self._quick_test(env_id="BipedalWalker-v3", algo="sac", train_reward="BHNR")

    def test_run_sac_bipedalwalker_hparams_default(self):
        hparams_file = self._bw_hparams_file
        self._quick_test(
            env_id="BipedalWalker-v3",
            algo="sac",
            train_reward="default",
            hparams_file=hparams_file,
        )

    def test_run_sac_lunarlander_default(self):
        self._quick_test(
            env_id="LunarLanderContinuous-v2", algo="sac", train_reward="default"
        )

    def test_run_sac_lunarlander_HPRS(self):
        self._quick_test(
            env_id="LunarLanderContinuous-v2", algo="sac", train_reward="HPRS"
        )

    def test_run_sac_lunarlander_TLTL(self):
        self._quick_test(
            env_id="LunarLanderContinuous-v2", algo="sac", train_reward="TLTL"
        )

    def test_run_sac_lunarlander_BHNR(self):
        self._quick_test(
            env_id="LunarLanderContinuous-v2", algo="sac", train_reward="BHNR"
        )

    def test_run_sac_lunarlander_hparams_default(self):
        hparams_file = self._ll_hparams_file
        self._quick_test(
            env_id="LunarLanderContinuous-v2",
            algo="sac",
            train_reward="default",
            hparams_file=hparams_file,
        )
