"""
Benchmark various reward shaping methods on the CartPole-v1 environment using Stable Baselines 3.

For each shaping method, we train a PPO agent for 50k timesteps and evaluate it every 1k timesteps.
The evaluation is done on the same (default) evaluation environment, so the evaluation metric is
the same for all methods.

The results are saved to the `exp` directory and the learning curves saved to `exp/learning_curves.png`.
"""

import numpy as np
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

import shaping

seed = 42
env_id = "CartPole-v1"
total_timesteps = 50_000
eval_freq = 1000
outdir = "exp"

# create eval env with the default environment
# pass through wrap() instead of directly gym.make() to ensure same obs/act spaces
eval_env = shaping.wrap(env=env_id, reward="default", env_kwargs={"render_mode": None})
eval_env = FlattenObservation(eval_env)

# train
results = {}
for reward in ["default", "TLTL", "HPRS"]:
    print(f"Training with {reward} reward shaping")

    # seed for reproducibility
    eval_env.reset(seed=seed)
    set_random_seed(seed)

    # create train env with the desired reward
    train_env = shaping.wrap(env=env_id, reward=reward)
    train_env = FlattenObservation(train_env)

    # train model and save results to logdir
    logdir = f"{outdir}/PPO-{env_id}-{reward}"
    eval_callback = EvalCallback(eval_env, log_path=logdir, eval_freq=eval_freq)
    model = PPO("MlpPolicy", train_env, seed=seed, tensorboard_log=logdir, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # read results from logdir
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
