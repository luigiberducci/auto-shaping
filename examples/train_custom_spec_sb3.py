import gymnasium
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C

import shaping
from shaping import RewardSpec

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")

specs = [
    'ensure "x" < 2.4',
    'ensure "x" > -2.4',
    'ensure "theta" < 0.2',
    'ensure "theta" > -0.2',
]
variables = [
    ("x", -2.4, 2.4),
    ("x_dot", -3.0, 3.0),
    ("theta", -0.2, 0.2),
    ("theta_dot", -3.0, 3.0),
]
spec = RewardSpec(specs=specs, variables=variables)

env = shaping.wrap(env=env, reward="TLTL", spec=spec)
env = FlattenObservation(env)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
