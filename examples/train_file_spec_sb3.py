import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C

import shaping

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = shaping.wrap(env=env, reward="TLTL", spec="../configs/CartPole-v1.yaml")
env = FlattenObservation(env)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
