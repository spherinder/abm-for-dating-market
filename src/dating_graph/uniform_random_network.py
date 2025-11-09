import networkx as nx
from ..agents.base import Agent
from dating_network import DatingNetwork

class UniformRandomNetwork(DatingNetwork):
    def __init__(self, p: float):
        """
        :param p: probability for each edge formation
        """
        super().__init__()
        self.p = p
    
    def build(self, agents: list[Agent]):
        n = len(agents)
        self.graph = nx.erdos_renyi_graph(n, self.p)
        for agent in agents:
            self.agent_lookup_dict[agent.id] = agent
            self.update_person_popularity(agent)        
