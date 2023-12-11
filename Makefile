run-cartpole:
	python3.10 run_benchmark.py --algo ppo --env-id CartPole-v1 --total-timesteps 100000 --hparams-file examples/hparams/ppo-CartPole-v1.yaml

run-bipedal-walker:
	python3.10 run_benchmark.py --algo sac --env-id BipedalWalker-v3 --total-timesteps 1000000 --hparams-file examples/hparams/sac-BipedalWalker-v3.yaml

run-lunar-lander:
	python3.10 run_benchmark.py --algo sac --env-id LunarLanderContinuous-v2 --total-timesteps 1000000 --hparams-file examples/hparams/sac-LunarLanderContinuous-v2.yaml