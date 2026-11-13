import numpy as np

from .meta_sampler import MetaSampler

class UniformSampler(MetaSampler):
    """Uniform Task Sampler that randomly selects a task class
    from the pool of tasks.
    """
    def _select_training_envs(self) -> list[str]:
        """Randomly pick a tasks to train on with equal probability"""
        classes = list(self._envs_by_class.keys())
        sampled = np.random.choice(
            classes, size=self.meta_batch_size, replace=True
        ).tolist()
        return sampled