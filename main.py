from collections.abc import Iterable
import random
import numpy as np

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from sortedcontainers import SortedSet
from rustworkx import AllPairsPathMapping, PyGraph, all_pairs_dijkstra_shortest_paths, undirected_gnp_random_graph

from src.utils import gumbel_weighted_permutation

Gender: TypeAlias = Literal["M", "F"]

@dataclass
class Agent:
    attr: float
    sought: float
    gender: Final[Gender]

    @classmethod
    def new(cls, g: Gender, attr_max: int):
        return cls(random.randint(0, attr_max), random.randint(0, attr_max), g)

class UniformPairer:
    def pair_up(self, num_m: int, num_f: int) -> Iterable[tuple[int,int]]:
        maleixs = list(range(num_m))
        femixs = list(range(num_f))

        if num_m > num_f:
            maleixs = random.sample(maleixs, num_f)
        elif num_m < num_f:
            femixs = random.sample(femixs, num_m)

        random.shuffle(femixs)
        return zip(maleixs, femixs)

class NetworkPairer:
    graph: Final[PyGraph[int]]
    paths: Final[AllPairsPathMapping]

    def __init__(self, n_agents: int, density: float):
        self.graph = undirected_gnp_random_graph(n_agents, density)
        self.paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)

    def pair_up(self, males: SortedSet[int], fems: SortedSet[int]) -> Iterable[tuple[int,int]]:
        num_m = len(males)
        num_f = len(fems)
        maleixs = list(range(num_m))
        femixs = list(range(num_f))

        if num_m > num_f:
            maleixs = random.sample(maleixs, num_f)
        elif num_m < num_f:
            femixs = random.sample(femixs, num_m)
        pathmaps = (self.paths[males[mi]] for mi in maleixs)
        weights = np.array([
            [ (len(pathmap[fems[fi]]) if fems[fi] in pathmap else 0) for fi in femixs ]
            for pathmap in pathmaps
        ])

        perm = gumbel_weighted_permutation(weights)
        return zip(maleixs, (femixs[int(i)] for i in perm))


@dataclass
class Simulation:
    agents: Final[list[Agent]]
    males: SortedSet[int]
    fems: SortedSet[int]
    couples: list[tuple[int,int]]
    attr_max: Final[int]
    T: Final[int]
    affection_malleability: Final[float] # ∈ [0,1]

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        return (self.attr_max - abs(a_i.sought - a_j.attr)) / self.attr_max

    def rejects(self, a_i: Agent, a_j: Agent):
        a_j.sought += self.affection_malleability * (a_j.sought - a_i.attr)

    def couple(self, m: int, f: int):
        self.males.remove(m)
        self.fems.remove(f)
        self.couples.append((m,f))

    def run_unif(self, pairer: UniformPairer):
        for _ in range(self.T):
            pairs = (
                (self.males[mi], self.fems[fi])
                for mi,fi in pairer.pair_up(len(self.males), len(self.fems))
            )

            pairs = list(pairs)
            print(f"got pairs {pairs}")

            for male, fem in pairs:
                a_m = self.agents[male]
                a_f = self.agents[fem]
                m_accepts = random.random() < self.accept_prob(a_m, a_f)
                f_accepts = random.random() < self.accept_prob(a_f, a_m)
                if m_accepts and f_accepts:
                    self.couple(male, fem)
                elif not m_accepts:
                    self.rejects(a_m,a_f)
                elif not f_accepts:
                    self.rejects(a_f,a_m)

    def run_network(self, pairer: NetworkPairer):
        for _ in range(self.T):
            if len(self.males) == 0 or len(self.fems) == 0:
                print(f"no more {"males" if len(self.males) == 0 else "females"}")
                break

            pairs = [
                (self.males[mi], self.fems[fi])
                for mi,fi in pairer.pair_up(self.males, self.fems)
            ]

            for male, fem in pairs:
                a_m = self.agents[male]
                a_f = self.agents[fem]
                m_accepts = random.random() < self.accept_prob(a_m, a_f)
                f_accepts = random.random() < self.accept_prob(a_f, a_m)
                if m_accepts and f_accepts:
                    self.couple(male, fem)
                elif not m_accepts:
                    self.rejects(a_m,a_f)
                elif not f_accepts:
                    self.rejects(a_f,a_m)

def main():
    N = 10
    agents = [Agent.new("M", 50) for _ in range(N)]
    agents.extend(Agent.new("F", 50) for _ in range(N))
    males = SortedSet(range(N))
    fems = SortedSet(range(N, N+N))
    sim = Simulation(
        agents, males, fems, [],
        50, 20, 0.1
    )

    print("initial state")
    import pprint
    pprint.pprint(sim)

    # pairer = UniformPairer()
    # sim.run_unif(pairer)
    pairer = NetworkPairer(N*2, 0.2)
    sim.run_network(pairer)

    print(f"got result")
    pprint.pprint(sim)

if __name__ == "__main__":
    main()
