import random
from abc import ABC, abstractmethod
from typing import Callable, Optional


class BasePolicy(ABC):
    """Interface expected by AStarSearch."""

    @abstractmethod
    def top_k_actions(self, obs, k: int) -> list:
        """Return a list of up to k action indices to expand from obs."""
        ...

    @abstractmethod
    def heuristic(self, obs) -> float:
        """Estimated cost-to-goal from obs (must be admissible for A* optimality)."""
        ...


class RandomPolicy(BasePolicy):
    """
    Baseline policy for A* search.

    top_k_actions — uniform random sample of action indices (no learning).
    heuristic     — always 0.0 (admissible; reduces A* to uniform-cost search).

    This is useful for verifying the search loop, benchmarking, and generating
    training data before a neural policy is available.
    """

    def __init__(self, n_actions: int, seed: int = None):
        self.n_actions = n_actions
        if seed is not None:
            random.seed(seed)

    def top_k_actions(self, obs, k: int) -> list:
        k = min(k, self.n_actions)
        return random.sample(range(self.n_actions), k)

    def heuristic(self, obs) -> float:
        return 0.0


class NeuralHeuristicPolicy(BasePolicy):
    """
    Policy that pairs random action selection with a trained HeuristicModel.

    Replaces the flat h=0 of RandomPolicy with a learned h(s), giving A*
    a meaningful priority signal while the policy network is not yet trained.

    Args:
        n_actions:        total number of actions in the environment.
        heuristic_fn:     callable obs -> float produced by
                          HeuristicModel.as_heuristic(env.obs_to_vector).
        seed:             optional random seed for action sampling.
    """

    def __init__(
        self,
        n_actions: int,
        heuristic_fn: Callable,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self._heuristic_fn = heuristic_fn
        if seed is not None:
            random.seed(seed)

    def top_k_actions(self, obs, k: int) -> list:
        k = min(k, self.n_actions)
        return random.sample(range(self.n_actions), k)

    def heuristic(self, obs) -> float:
        return self._heuristic_fn(obs)
