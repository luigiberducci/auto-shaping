import os
import time
from typing import Any

import numpy as np
import torch
from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

import logging


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        video_folder: str = None,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param video_folder: The directory in which to save the video, otherwise defaults to the TensorBoard
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._video_folder = video_folder
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

        # create video folder if it does not exist
        if self._video_folder is not None:
            os.makedirs(self._video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            t0 = time.time()


            if self._video_folder is not None:
                video_name = f"video-step-{self.n_calls}"
                base_path = os.path.join(self._video_folder, video_name)
            else:
                base_path = None

            self.video_recorder = video_recorder.VideoRecorder(
                env=self._eval_env,
                base_path=base_path,
                metadata={"call_id": self.n_calls},
            )

            def grab_screen(_locals: dict, _globals: dict) -> None:
                self.video_recorder.capture_frame()

            rewards, lengths = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screen,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            # grab last frame
            self.video_recorder.capture_frame()
            self.video_recorder.metadata["episode_reward"] = rewards
            self.video_recorder.metadata["episode_length"] = lengths
            self.video_recorder.close()

            t1 = time.time()
            logging.debug(f"Video recording took {t1-t0} seconds")

        return True
