import random
from typing import Protocol
from .base import *

class RandomUniformMatching(MatchmakingModel):
    def match(self, agents: list[Agent]) -> list[tuple[Agent, Agent]]:
        """A simple matchmaking strategy that pairs agents sequentially."""
        agents_copy = agents[:]
        pairs = []
        while len(agents_copy) >= 2: 
            index1 = random.randint(0, len(agents_copy) - 1)
            index2 = random.randint(0, len(agents_copy) - 1)
            agents_copy[index1], agents_copy[0] = agents_copy[0], agents_copy[index1]
            agents_copy[index2], agents_copy[1] = agents_copy[1], agents_copy[index2]
            pairs.append((agents_copy[0], agents_copy[1]))
            agents_copy = agents_copy[2:]
        return pairs