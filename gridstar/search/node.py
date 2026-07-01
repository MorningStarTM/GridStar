from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchNode:
    """
    One node in the A* search tree.

    `obs`        — grid2op observation at this state
    `g`          — accumulated cost from the root
    `h`          — heuristic estimate of cost to goal
    `parent`     — parent SearchNode (None for root)
    `action_idx` — integer action index taken from parent (-1 for root)
    `depth`      — tree depth (root = 0)
    `node_id`    — unique monotonic id; used as heap tiebreaker
    """

    obs: object
    g: float
    h: float
    parent: Optional[SearchNode]
    action_idx: int
    depth: int
    node_id: int

    @property
    def f(self) -> float:
        return self.g + self.h

    # Heap comparison: when f values tie, expand the node inserted earlier.
    def __lt__(self, other: SearchNode) -> bool:
        return self.node_id < other.node_id
