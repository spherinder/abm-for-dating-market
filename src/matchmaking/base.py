from typing import Protocol
from ..agents.simple import Agent
from ..dating_graph.dating_graph import DatingGraph

class MatchmakingModel(Protocol):
    def match(self, agents: DatingGraph) -> list[tuple[Agent, Agent]]:
        """Match agents according to the model's strategy."""
        raise NotImplementedError("Subclasses should implement this method.")
