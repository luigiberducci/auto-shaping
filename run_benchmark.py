import os
import logging
from argparse import Namespace

import numpy as np

import run

logging.basicConfig(level=logging.INFO)
os.environ["WANDB_MODE"] = "offline"

config = {
    "algo": "ppo",
    "env_id": "CartPole-v1",
    "total_timesteps": 1e5,
    "eval_frequency": 1e4,
    "hparams_file": None,
    "train_reward": None,  # set in loop
    "eval_reward": "default",
    "spec_file": None,
    "log_dir": "logs",
    "wandb": False,
    "wandb_entity": None,
    "wandb_project": None,
    "wandb_group": None,
    "seed": 0,
}

# run benchmark
for reward in ["default", "HPRS", "TLTL", "BHNR"]:
    print(f"Training with {reward} reward shaping")

    cfg = config.copy()
    cfg["train_reward"] = reward
    run.main(args=Namespace(**cfg))

# load results from logdirectory
# todo
results = {}

# plot learning curves
if len(results) > 0:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title(f"Learning Curves - {cfg['env_id']} - {cfg['algo']}")

    for reward in results:
        rewards = results[reward]["rewards"]
        means, stds = np.mean(rewards, axis=1), np.std(rewards, axis=1)
        plt.plot(results[reward]["timesteps"], means, label=reward)
        plt.fill_between(
            results[reward]["timesteps"], means - stds, means + stds, alpha=0.2
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")

    # place legend outside plot in bottom center
    plt.subplots_adjust(bottom=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc="lower center", ncol=len(results))

    plt.savefig(config["log_dir"] + "/learning_curves.png")
    plt.show()
