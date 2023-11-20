import gymnasium
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import A2C

from shaping import Variable
from shaping.tltl_shaping import TLTLWrapper

RENDER=False    # Set to True to render the environment after training

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")

specs = [
    'ensure abs "x" < 2.4',
    'ensure abs "theta" < 0.2',
]
variables = [
    Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
    Variable(name="x_dot", fn="state[1]",  min=-3.0, max=3.0),
    Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
    Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
]

env = TLTLWrapper(env, specs=specs, variables=variables)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if RENDER:
        vec_env.render("human")

