import pathlib
from datetime import datetime

import numpy as np
import yaml
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

import shaping
from training.schedules import linear_schedule
from training.video_recorder_cb import VideoRecorderCallback

ALGOS = {"ppo": PPO, "sac": SAC}
REWARDS = ["default", "TLTL", "BHNR", "HPRS"]


def load_hparams(file: str):
    assert pathlib.Path(file).exists(), f"file {file} does not exist"

    with open(file, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    for k in ["learning_rate", "clip_range"]:
        if k in hparams and isinstance(hparams[k], str):
            fn, val = hparams[k].split("_")
            assert fn == "lin", f"only linear schedules are supported, got {fn}"
            hparams[k] = linear_schedule(float(val))

    return hparams


def main(args):
    callbacks = []

    # logging
    exp_name = f"{args.algo}-{args.env_id}-{args.train_reward}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"{args.log_dir}/{exp_name}/run-{timestamp}"

    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    if args.wandb:
        assert args.wandb_entity is not None, "must provide wandb entity"
        assert args.wandb_project is not None, "must provide wandb project"
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=exp_name,
            group=args.wandb_group,
            config=dict(args),
        )

        wand_callback = WandbCallback(model_save_path=f"{logdir}/models", verbose=2)
        callbacks.append(wand_callback)

    # set up environments
    seed = args.seed or np.random.randint(0, 2 ** 32 - 1)
    set_random_seed(seed)

    env_kwargs = {"render_mode": "rgb_array"}
    train_env = shaping.wrap(
        env=args.env_id,
        reward=args.train_reward,
        spec=args.spec_file,
        env_kwargs=env_kwargs,
    )
    train_env = FlattenObservation(train_env)

    eval_env = shaping.wrap(
        env=args.env_id,
        reward=args.eval_reward,
        spec=args.spec_file,
        env_kwargs=env_kwargs,
    )
    eval_env = FlattenObservation(eval_env)
    eval_env = Monitor(eval_env)

    # training callbacks
    eval_callback = EvalCallback(
        eval_env, log_path=logdir, eval_freq=args.eval_frequency
    )

    video_callback = VideoRecorderCallback(
        eval_env,
        n_eval_episodes=1,
        render_freq=args.eval_frequency,
        video_folder=f"{logdir}/videos",
    )
    callbacks.append(eval_callback)
    callbacks.append(video_callback)

    # set up model

    if args.hparams_file is not None:
        hparams = load_hparams(args.hparams_file)
    else:
        hparams = {"policy": "MlpPolicy"}

    algo_cls = ALGOS[args.algo]
    model = algo_cls(
        env=train_env, seed=args.seed, tensorboard_log=logdir, verbose=1, **hparams
    )

    # train
    model.learn(
        total_timesteps=args.total_timesteps, callback=callbacks,
    )

    if args.wandb:
        run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a reward shaping experiment.")

    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="The registered environment for training the agent.",
    )
    parser.add_argument(
        "--spec-file",
        type=str,
        default=None,
        help="The file containing the specification. If not provided, "
        "it looks for a file in configs/ with the same name as the environment.",
    )
    parser.add_argument(
        "--train-reward",
        type=str,
        default="default",
        choices=REWARDS,
        help="The reward shaping to use.",
    )
    parser.add_argument(
        "--eval-reward",
        type=str,
        default="default",
        choices=REWARDS,
        help="The reward shaping to use.",
    )

    parser.add_argument("--seed", type=int, default=0, help="The random seed to use.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1e5,
        help="The total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10000,
        help="The frequency at which to evaluate the agent.",
    )

    parser.add_argument("--wandb", action="store_true", help="Whether to log to wandb.")
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="The wandb entity to log to."
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The wandb project to log to."
    )
    parser.add_argument(
        "--wandb-group-id", type=str, default=None, help="The wandb group id to log to."
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/", help="The local directory to log to."
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=list(ALGOS.keys()),
        help="The RL algorithm to use.",
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        default=None,
        help="The file containing the algorithm configuration.",
    )

    args = parser.parse_args()
    main(args)
