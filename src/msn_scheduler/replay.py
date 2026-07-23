from __future__ import annotations

import random
from collections import deque

from .datatypes import Transition


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.data):
            raise ValueError("Not enough replay samples")
        return random.sample(list(self.data), batch_size)
