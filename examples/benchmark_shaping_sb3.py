import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

import shaping

env_id = "CartPole-v1"
total_timesteps = 25_000

# create eval env, pass through wrap() to ensure same obs/act spaces
eval_env = shaping.wrap(env=env_id, reward="default", env_kwargs={"render_mode": None})
eval_env = FlattenObservation(eval_env)

# train
results = {}
for reward in ["default", "TLTL", "HPRS"]:
    print(f"Training with {reward} reward shaping")
    train_env = shaping.wrap(env=env_id, reward=reward)
    train_env = FlattenObservation(train_env)

    model = A2C("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    # evaluate
    rewards, lengths = evaluate_policy(
        model, eval_env, n_eval_episodes=10, render=True, return_episode_rewards=True
    )
    results[reward] = {
        "reward": {"mean": np.mean(rewards), "std": np.std(rewards),},
        "length": {"mean": np.mean(lengths), "std": np.std(lengths),},
    }

    print(f"Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Length: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

# print results
print("\nResults:")
for reward, result in results.items():
    print(
        f"{reward}: reward: {result['reward']['mean']:.2f} +/- {result['reward']['std']:.2f}, "
        f"length: {result['length']['mean']:.2f} +/- {result['length']['std']:.2f}"
    )
