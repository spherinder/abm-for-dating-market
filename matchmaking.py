from typing import Protocol
from agent import Agent

class MatchmakingModel(Protocol):
    def match(self, agent1: Agent, agent2: Agent):
        """Match agents according to the model's strategy."""
        pass


class SimpleMatchmakingModel(MatchmakingModel):
    def match(self, agent1: Agent, agent2: Agent):
        """A simple matchmaking strategy that pairs agents sequentially."""
        return abs(agent1.a - agent2.u) < 10 and abs(agent2.a - agent1.u) < 10