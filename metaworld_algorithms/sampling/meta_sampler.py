
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable

import gymnasium as gym
import metaworld

from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.types import Task


class MetaSampler(ABC):
    """Base class for sampling MetaWorld training environments.

    Creates and stores all training environments from a MetaWorld
    benchmark. Subclasses implement `_select_training_envs` to define
    how environment classes are selected.

    Tasks within each environment class are selected sequentially,
    wrapping around when the end of the task list is reached.
    """

    def __init__(self, benchmark: metaworld.Benchmark) -> None:
        self._envs_by_class: dict[str, list[SawyerXYZEnv]] = (
            self._create_training_envs(benchmark)
        )

        # Tracks the next task to select for each environment class.
        self._next_task_index_by_class = {
            class_name: 0 for class_name in self._envs_by_class
        }

    @staticmethod
    def _make_env(
        env_cls: type[SawyerXYZEnv],
        task: Task
    ) -> SawyerXYZEnv:
        """Create a MetaWorld environment and assign its task."""
        env = env_cls()
        env.set_task(task)
        return env

    @staticmethod
    def _to_vec_env(
        envs: list[gym.Env],
    ) -> gym.vector.SyncVectorEnv:
        """Wrap existing environments in a synchronous vector environment."""
        env_fns: list[Callable[[], gym.Env]] = [
            lambda env=env: env for env in envs
        ]
        return gym.vector.SyncVectorEnv(env_fns)

    @classmethod
    def _create_training_envs(
        cls,
        benchmark: metaworld.Benchmark
    ) -> dict[str, list[SawyerXYZEnv]]:
        """Create all training environments grouped by environment class."""
        envs_by_class: dict[str, list[SawyerXYZEnv]] = defaultdict(list)

        for task in benchmark.train_tasks:
            env_cls = benchmark.train_classes[task.env_name]
            env = cls._make_env(env_cls, task)
            envs_by_class[task.env_name].append(env)

        return dict(envs_by_class)

    @abstractmethod
    def _select_training_envs(self, num_envs: int) -> list[str]:
        """Select environment classes for the next training batch."""
        raise NotImplementedError

    def sample(self, num_envs: int = 20) -> gym.vector.SyncVectorEnv:
        """Sample a batch of training environments.

        Environment classes are selected by `_select_training_envs`.
        For each selected class, the next available task is used.
        Task selection wraps around when all tasks of a class have
        been selected.
        """
        selected_classes = self._select_training_envs(num_envs)

        assert len(selected_classes) == num_envs, (
            f"_select_training_envs returned {len(selected_classes)} "
            f"classes, but {num_envs} were requested."
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
