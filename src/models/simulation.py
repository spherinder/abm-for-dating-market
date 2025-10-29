from typing import List
import numpy as np
from ..agents.base import Agent
from ..matchmaking.base import MatchmakingModel

class Simulation:
    def __init__(
        self,
        agents: List[Agent],
        matchmaking_model: MatchmakingModel,
        T: int = 50,
    ):
        self.agents = agents
        self.matching_model = matchmaking_model
        self.T = T # rounds of simulation
        self.D = np.zeros((len(agents), len(agents)))  # dating matrix history (can be simplified by addressing gender)

    def run(self):
        for t in range(self.T):
            pairs: list[tuple[Agent, Agent]] = self.matching_model.match(self.agents)

            for agent_i, agent_j in pairs:
                if agent_i.decide_match(agent_j) and agent_j.decide_match(agent_i):
                    # they date
                    agent_i.d= agent_j.d = True
                    self.D[agent_i.id, agent_j.id] = self.D[agent_j.id, agent_i.id] = 1
                else:
                    # rejected – optionally update willingness or u drift
                    agent_i.handle_rejection(agent_j)
                    agent_j.handle_rejection(agent_i)
                    agent_i.d = agent_j.d = False
                    pass

            # Here later: update willingness u, track exclusion, log state

        return self.D
