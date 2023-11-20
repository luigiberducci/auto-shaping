import logging
import pathlib
import re
from argparse import Namespace

import numpy as np

import run

logging.basicConfig(level=logging.INFO)

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
algo, env_id = config["algo"], config["env_id"]

rewards = ["default", "HPRS", "TLTL", "BHNR"]
for reward in rewards:
    print(f"Training with {reward} reward shaping")

    cfg = config.copy()
    cfg["train_reward"] = reward
    cfg["wandb_group"] = f"{algo}-{env_id}-{reward}"
    run.main(args=Namespace(**cfg))

# load results from logdirectory
results = {}
for reward in rewards:
    eval_files = list(
        pathlib.Path(config["log_dir"]).glob(
            f"**/{algo}-{env_id}-{reward}/*/evaluations.npz"
        )
    )

    if len(eval_files) > 1:
        logging.warning(
            f"Found {len(eval_files)} eval files for {algo}-{env_id}-{reward}. Plotting only the 1st one."
        )

    file = eval_files[0]
    if len(re.findall(f"{algo}-{env_id}-{reward}", str(file.parent))) > 0:
        data = np.load(file, allow_pickle=True)
        results[reward] = {
            "timesteps": data["timesteps"],
            "rewards": data["results"],
            "lengths": data["ep_lengths"],
        }

# plot learning curves
if len(results) > 0:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title(f"Learning Curves - {env_id} - {algo}")

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
