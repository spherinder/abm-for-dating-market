from ..agents.base import Agent

class DatingGraph:
    def __init__(self):
        self.graph : dict[Agent, set[Agent]] = {}

    def add_person(self, person: Agent):
        if person not in self.graph:
            self.graph[person] = set()

    def add_connection(self, person1: Agent, person2: Agent):
        self.add_person(person1)
        self.add_person(person2)
        self.graph[person1].add(person2)
        self.graph[person2].add(person1)

    def remove_connection(self, person1: Agent, person2: Agent):
        if person1 in self.graph and person2 in self.graph[person1]:
            self.graph[person1].remove(person2)
        if person2 in self.graph and person1 in self.graph[person2]:
            self.graph[person2].remove(person1)

    def get_connections(self, person: Agent) -> set[Agent]:
        return self.graph.get(person, set())
    
    def to_agents(self) -> list[Agent]:
        agents = []
        for person in self.graph:
            agent = Agent(id=person.id)
            agents.append(agent)
        return agents
    
    def from_agents(self, agents: list[Agent]):
        for agent in agents:
            self.add_person(agent)

    def complete_graph(self):
        for person in self.graph:
            connections = self.get_connections(person)
            for other in self.graph:
                if other != person and other not in connections and person.g != other.g:
                    self.add_connection(person, other)

    def random_uniform_edges(self, probability: float):
        import random
        for person in self.graph:
            for other in self.graph:
                if other != person and random.random() < probability and person.g != other.g:
                    self.add_connection(person, other)