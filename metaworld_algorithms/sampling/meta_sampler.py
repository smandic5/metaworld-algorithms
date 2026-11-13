
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Literal
import jax
import numpy as np

import gymnasium as gym
import metaworld

from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.types import Task
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    RNNBasedMetaRLWrapper,
    NormalizeRewardsExponential,
)

from metaworld_algorithms.rl.algorithms.utils import MetaTrainState, TrainState


class MetaSampler(ABC):
    """Base class for sampling MetaWorld training environments.

    Creates and stores all training environments from a MetaWorld
    benchmark. Subclasses implement `_select_training_envs` to define
    how environment classes are selected.

    Tasks within each environment class are selected sequentially,
    wrapping around when the end of the task list is reached.
    """
    reward_func_version: str = "v2"
    num_goals: int = 50
    reward_normalization_method: str | None = None
    normalize_observations: bool = False
    render_mode: Literal["rgb_array"] = "rgb_array"
    camera_name: str | None = "corner"
    camera_id: int | None = None
    
    max_episode_steps: int = 500
    terminate_on_success: bool = False
    recurrent_info_in_obs: bool = False
    
    use_one_hot: bool = False
    meta_batch_size: int = 20

    total_goals_per_task_train: int = 50
    total_goals_per_task_test: int = 40

    evaluation_num_episodes: int = 3
    evaluation_adaptation_steps: int = 1
    evaluation_adaptation_episodes: int = 10

    def __init__(self, benchmark: metaworld.Benchmark, seed: int) -> None:
        self.seed = seed
        self._envs_by_class: dict[str, list[gym.Env]] = (
            self._create_training_envs(benchmark)
        )

        # Tracks the next task to select for each environment class.
        self._next_task_index_by_class = {
            class_name: 0 for class_name in self._envs_by_class
        }

    def _make_env(
        self,
        env_cls: type[SawyerXYZEnv],
        task: Task
    ) -> gym.Env:
        """Create a MetaWorld environment and assign its task."""
        # Underlying metaworld environment
        env = env_cls(reward_function_version=self.reward_func_version)
        env.set_task(task)
        # wrappers
        env = gym.wrappers.TimeLimit(env, self.max_episode_steps)
        env = AutoTerminateOnSuccessWrapper(env)
        env.toggle_terminate_on_success(self.terminate_on_success)
        if self.recurrent_info_in_obs:
            env = RNNBasedMetaRLWrapper(env)
        if self.reward_normalization_method == "gymnasium":
            env = gym.wrappers.NormalizeReward(env)
        elif self.reward_normalization_method == "exponential":
            env = NormalizeRewardsExponential(env=env, reward_alpha = 0.001)
        if self.normalize_observations:
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # seed
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)

        return env

    @staticmethod
    def _to_vec_env(
        envs: list[gym.Env],
    ) -> gym.vector.SyncVectorEnv:
        """Wrap existing environments in a synchronous vector environment."""
        env_fns: list[Callable[[], gym.Env]] = [
            lambda env=env: env for env in envs
        ]
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    def _create_training_envs(
        self,
        benchmark: metaworld.Benchmark
    ) -> dict[str, list[gym.Env]]:
        """Create all training environments grouped by environment class."""
        envs_by_class: dict[str, list[gym.Env]] = defaultdict(list)

        for task in benchmark.train_tasks:
            env_cls = benchmark.train_classes[task.env_name]
            env = self._make_env(env_cls, task)
            envs_by_class[task.env_name].append(env)

        return dict(envs_by_class)

    @abstractmethod
    def _select_training_envs(self) -> list[str]:
        """Select environment classes for the next training batch."""
        raise NotImplementedError

    def sample(self) -> gym.vector.SyncVectorEnv:
        """Sample a batch of training environments.

        Environment classes are selected by `_select_training_envs`.
        For each selected class, the next available task is used.
        Task selection wraps around when all tasks of a class have
        been selected.
        """
        selected_classes = self._select_training_envs()

        assert len(selected_classes) == self.meta_batch_size, (
            f"_select_training_envs returned {len(selected_classes)} "
            f"classes, but {self.meta_batch_size} were requested."
        )

        selected_envs = []

        for class_name in selected_classes:
            available_envs = self._envs_by_class[class_name]

            assert available_envs, (
                f"No training environments available for '{class_name}'."
            )

            # Select the next task and advance the index.
            task_index = self._next_task_index_by_class[class_name]
            selected_envs.append(available_envs[task_index])

            self._next_task_index_by_class[class_name] = (
                task_index + 1
            ) % len(available_envs)

        return self._to_vec_env(selected_envs)

    def extract_ensemble_params(self, params):
        params = jax.device_get(params)
        return [
            jax.tree_util.tree_map(
                lambda x: np.array(x[i], copy=True),
                params
            )
            for i in range(self.meta_batch_size)
        ]

    def feedback(
        self,
        train_state: TrainState,
        observations: np.ndarray,
    ) -> None:
        params = self.extract_ensemble_params(train_state.params)