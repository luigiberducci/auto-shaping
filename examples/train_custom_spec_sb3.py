import gymnasium as gym

from stable_baselines3 import A2C

from shaping.spec.reward_spec import RewardSpec
from shaping.tltl_shaping import TLTLWrapper

env = gym.make("CartPole-v1", render_mode="rgb_array")
spec = RewardSpec(
            specs=[
                "always(x < 2.4)",
                "always(x > -2.4)",
                "always(theta < 0.2)",
                "always(theta > -0.2)",
            ],
            variables=["x", "x_dot", "theta", "theta_dot"],
            ranges=[(-2.4, 2.4), (-3.0, 3.0), (-0.2, 0.2), (-3.0, 3.0)],
        )
env = TLTLWrapper(env, spec)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")