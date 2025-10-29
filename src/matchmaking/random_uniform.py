import random
from typing import Protocol
from .base import *

class RandomUniformMatching(MatchmakingModel):
    def match(self, agents: list[Agent]) -> list[tuple[Agent, Agent]]:
        """A simple matchmaking strategy that pairs agents sequentially."""
        free_agents = [a for a in agents if not a.d]
        pairs = []
        while len(free_agents) >= 2: 
            index1 = random.randint(0, len(free_agents) - 1)
            index2 = random.randint(0, len(free_agents) - 1)
            # swap to front
            free_agents[index1], free_agents[0] = free_agents[0], free_agents[index1]
            free_agents[index2], free_agents[1] = free_agents[1], free_agents[index2]
            pairs.append((free_agents[0], free_agents[1]))
            free_agents = free_agents[2:]
        return pairs