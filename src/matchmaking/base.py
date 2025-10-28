from typing import Protocol
from agents.simple_agent import Agent

class MatchmakingModel(Protocol):
    def match(self, agents: list[Agent]) -> list[tuple[Agent, Agent]]:
        """Match agents according to the model's strategy."""
        pass
