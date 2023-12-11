import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C

import auto_shaping

RENDER = False  # Set to True to render the environment after training

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = auto_shaping.wrap(env=env, reward="HPRS", spec="../configs/CartPole-v1.yaml")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if RENDER:
        vec_env.render("human")
