import random
from typing import Protocol
from .base import *

class RandomUniformPairing(PairingModel):
    def match(self, agents: DatingGraph) -> list[tuple[Agent, Agent]]:
        """A simple matchmaking strategy that pairs agents sequentially."""
        agents.complete_graph()
        pairs = []
        for agent in agents.graph.keys():
            connections = agents.get_connections(agent)
            free_connections = [c for c in connections if not any(c in p for p in pairs)]
            if len(free_connections) > 0:
                match_id = random.choice(free_connections)
                pairs.append((agent, match_id))
        return pairs

        # free_agents = [a for a in agents if not a.d]
        # pairs = []
        # while len(free_agents) >= 2: 
        #     index1 = random.randint(0, len(free_agents) - 1)
        #     index2 = random.randint(0, len(free_agents) - 1)
        #     # swap to front
        #     free_agents[index1], free_agents[0] = free_agents[0], free_agents[index1]
        #     free_agents[index2], free_agents[1] = free_agents[1], free_agents[index2]
        #     pairs.append((free_agents[0], free_agents[1]))
        #     free_agents = free_agents[2:]
        # return pairs