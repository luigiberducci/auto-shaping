"""
Benchmark various reward shaping methods on the CartPole-v1 environment using Stable Baselines 3.

For each shaping method, we train a PPO agent for 50k timesteps and evaluate it every 1k timesteps.
The evaluation is done on the same (default) evaluation environment, so the evaluation metric is
the same for all methods.

The results are saved to the `exp` directory and the learning curves saved to `exp/learning_curves.png`.
"""

import os
import time
import logging

from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

from training.callbacks.video_recorder_cb import VideoRecorderCallback

logging.basicConfig(level=logging.INFO)
os.environ["WANDB_MODE"] = "online"

import shaping

config = {
    "env_id": "CartPole-v1",
    "evaluation": {
        "reward": "default",
        "frequency": 2500,
    },
    "training": {
        "reward": None,  # set in loop
        "total_timesteps": 50000,
    },
    "outdir": "exp",
    "seed": 42,
}

# create eval env with the default environment
# pass through wrap() instead of directly gym.make() to ensure same obs/act spaces
eval_cfg = config["evaluation"]
eval_env = shaping.wrap(env=config["env_id"], reward=eval_cfg["reward"], env_kwargs={"render_mode": "rgb_array"})
eval_env = FlattenObservation(eval_env)

# train
results = {}
group_id = f"group-{int(time.time())}"
for reward in ["default", "TLTL", "BHNR", "HPRS"]:
    print(f"Training with {reward} reward shaping")

    # set up config
    cfg = config.copy()
    cfg["training"]["reward"] = reward

    # logging
    run = wandb.init(
        entity="luigiberducci",
        project="auto-shaping",
        name=f"{cfg['env_id']}-{reward}",
        group=group_id,
        config=cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # seed for reproducibility
    seed = cfg["seed"]
    eval_env.reset(seed=seed)
    set_random_seed(seed)

    # create train env with the desired reward
    train_cfg = cfg["training"]
    train_env = shaping.wrap(env=cfg["env_id"], reward=train_cfg["reward"])
    train_env = FlattenObservation(train_env)

    # train model and save results to logdir
    logdir = f"{cfg['outdir']}/group-{group_id}/PPO-{cfg['env_id']}-{reward}"
    eval_callback = EvalCallback(eval_env, log_path=logdir, eval_freq=eval_cfg["frequency"])
    wand_callback = WandbCallback(
        model_save_path=f"{logdir}/models",
        verbose=2,
    )
    video_callback = VideoRecorderCallback(eval_env, n_eval_episodes=1, render_freq=eval_cfg["frequency"],
                                           video_folder=f"{logdir}/videos")
    model = PPO("MlpPolicy", train_env, seed=seed, tensorboard_log=logdir, verbose=1)
    model.learn(total_timesteps=train_cfg["total_timesteps"],
                callback=[eval_callback, wand_callback, video_callback])

    run.finish()

    # read results from logdir
    """
    results[reward] = {}
    with open(f"{logdir}/evaluations.npz", "rb") as f:
        data = np.load(f)
        results[reward]["timesteps"] = data["timesteps"]
        results[reward]["rewards"] = data["results"]

    # evaluate trained model
    rewards, lengths = evaluate_policy(
        model, eval_env, n_eval_episodes=10, render=True, return_episode_rewards=True
    )

    print(f"Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Length: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    """

exit(0)
# plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.title(f"Learning Curves - {env_id} - PPO")

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

plt.savefig(outdir + "/learning_curves.png")
plt.show()
