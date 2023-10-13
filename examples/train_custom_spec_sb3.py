import gymnasium
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C

from shaping.tltl_shaping import TLTLWrapper
from shaping.utils.dictionary_wrapper import DictWrapper

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
env = DictWrapper(env, variables=["x", "x_dot", "theta", "theta_dot"])

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

env = TLTLWrapper(env, specs=specs, variables=variables)
env = FlattenObservation(env)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
