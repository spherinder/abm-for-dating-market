import networkx as nx
from ..agents.base import Agent
from dating_network import DatingNetwork

class BarabasiAlbertNetwork(DatingNetwork):
    def __init__(self, m: float):
        """
        :param m: number of edges to attach from a new node to existing nodes; the higher this number, the denser the graph
        """
        super().__init__()
        self.m = m
    
    def build(self, agents: list[Agent]):
        n = len(agents)
        self.graph = nx.barabasi_albert_graph(n, self.m)
        for agent in agents:
            self.agent_lookup_dict[agent.id] = agent
            self.update_person_popularity(agent) 
        