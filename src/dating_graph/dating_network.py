from networkx import Graph
from ..agents.base import Agent

class DatingNetwork:
    def __init__(self):
        self.graph : Graph = Graph()
        self.agent_lookup_dict: dict[int, Agent] = {}

    def add_person(self, person: Agent):
        if person.id not in self.graph:
            self.graph.add_node(person.id)
            self.agent_lookup_dict[person.id] = person

    def add_persons(self, agents: list[Agent]):
        for agent in agents:
            self.add_person(agent)

    def is_person_in_graph(self, person: Agent):
        return person.id in self.graph
    
    def remove_person(self, person: Agent):
        if self.is_person_in_graph(person):
            self.graph.remove_node(person.id)
            self.agent_lookup_dict.pop(person.id)

    def add_connection(self, person1: Agent, person2: Agent):
        if not self.is_person_in_graph(person1):
            self.add_person(person1)
        if not self.is_person_in_graph(person2):
            self.add_person(person2)
        self.graph.add_edge(person1.id, person2.id)

    def remove_connection(self, person1: Agent, person2: Agent):
        if self.is_person_in_graph(person1) and self.is_person_in_graph(person2):
            self.graph.remove_edge(person1.id, person2.id)

    def get_connections(self, person: Agent) -> set[Agent]:
        return set(self.graph.neighbors(person.id))
    
    def node_to_agent(self, node_id: int) -> Agent:
        return self.agent_lookup_dict[node_id]
    
    def update_person_popularity(self, person: Agent):
        person.pop = self.graph.degree[person.id]
    
    def build(self, agents: list[Agent]):
        raise NotImplementedError("Subclasses should implement this method.")
    
