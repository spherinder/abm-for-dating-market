from collections.abc import Iterable
from random import Random
import numpy as np
import pprint

from dataclasses import dataclass, field
from typing import Final, override
from sortedcontainers import SortedSet
from rustworkx import AllPairsPathMapping, PyGraph, all_pairs_dijkstra_shortest_paths, undirected_gnp_random_graph

from src.utils import gumbel_weighted_permutation

@dataclass
class Agent:
    attr: float
    sought: float
    rng: Final[Random] = field(repr=False)

    @classmethod
    def new(cls, attr_max: int, rng: Random):
        return cls(rng.randint(0, attr_max), rng.randint(0, attr_max), rng)

@dataclass
class UniformPairer:
    rng: Final[Random] = field(repr=False)

    def pair_up(self, males: SortedSet[int], fems: SortedSet[int]) -> Iterable[tuple[int,int]]:
        num_m = len(males)
        num_f = len(fems)
        maleixs = list(range(num_m))
        femixs = list(range(num_f))

        if num_m > num_f:
            maleixs = self.rng.sample(maleixs, num_f)
        elif num_m < num_f:
            femixs = self.rng.sample(femixs, num_m)

        self.rng.shuffle(femixs)
        return zip(maleixs, femixs)

class NetworkPairer:
    paths: Final[AllPairsPathMapping]
    rng: Final[Random]
    graph: Final[PyGraph[int]] # not necessary to store this
    density: Final[float] # not necessary to store this

    def __init__(self, n_agents: int, density: float, rng: Random):
        seed = rng.randint(0, 2**32 - 1)
        self.graph = undirected_gnp_random_graph(n_agents, density, seed)
        self.paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        self.rng = rng
        self.density = density

    def pair_up(self, males: SortedSet[int], fems: SortedSet[int]) -> Iterable[tuple[int,int]]:
        num_m = len(males)
        num_f = len(fems)
        maleixs = list(range(num_m))
        femixs = list(range(num_f))

        if num_m > num_f:
            maleixs = self.rng.sample(maleixs, num_f)
        elif num_m < num_f:
            femixs = self.rng.sample(femixs, num_m)
        pathmaps = (self.paths[males[mi]] for mi in maleixs)
        weights = np.array([
            [ (len(pathmap[fems[fi]]) if fems[fi] in pathmap else 0) for fi in femixs ]
            for pathmap in pathmaps
        ])

        perm = gumbel_weighted_permutation(weights)
        return zip(maleixs, (femixs[int(i)] for i in perm))

    def pretty_print(self, agents: list[Agent]) -> str:
        edges = "\n  ".join(
            ", ".join(f"{(agents[edge[0]], agents[edge[1]])}" for edge in self.graph.edge_list()[i:i+2])
            for i in range(0, len(self.graph.edge_list()), 5))
        return f"NetworkPairer(n_agents={len(self.graph)}, density={self.density}, Edges=[\n  {edges}\n])"



@dataclass
class Simulation:
    agents: Final[list[Agent]]
    males: SortedSet[int]
    fems: SortedSet[int]
    couples: list[tuple[int,int]]
    attr_max: Final[int]
    T: Final[int]
    affection_malleability: Final[float] # ∈ [0,1]
    rng: Final[Random] = field(repr=False)

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        return (self.attr_max - abs(a_i.sought - a_j.attr)) / self.attr_max

    def rejects(self, a_i: Agent, a_j: Agent):
        a_j.sought += self.affection_malleability * (a_j.sought - a_i.attr)

    def couple(self, m: int, f: int):
        self.males.remove(m)
        self.fems.remove(f)
        self.couples.append((m,f))

    def run(self, pairer: UniformPairer | NetworkPairer):
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
                m_accepts = self.rng.random() < self.accept_prob(a_m, a_f)
                f_accepts = self.rng.random() < self.accept_prob(a_f, a_m)
                if m_accepts and f_accepts:
                    self.couple(male, fem)
                elif not m_accepts:
                    self.rejects(a_m,a_f)
                elif not f_accepts:
                    self.rejects(a_f,a_m)

    @override
    def __repr__(self) -> str:
        males = "\n    ".join(f"{self.agents[a]}" for a in self.males)
        fems = "\n    ".join(f"{self.agents[a]}" for a in self.fems)
        couples = "\n    ".join((f"{(self.agents[m],self.agents[f])}" for m,f in self.couples))
        return (f"Simulation(attr_max={self.attr_max}, affection_malleability={self.affection_malleability}, T={self.T},"
           f"\n  males=[\n    {males}\n  ],\n  females=[\n    {fems}\n  ],\n  couples=[\n    {couples}\n  ]\n)"
           )

def main():
    SEED = 42
    rng = Random(SEED)

    N = 10 # agents per gender
    agents = [Agent.new(50, rng) for _ in range(N)]
    agents.extend(Agent.new(50, rng) for _ in range(N))
    males = SortedSet(range(N))
    fems = SortedSet(range(N, N+N))
    sim = Simulation(
        agents, males, fems, [],
        50, 20, 0.1, rng
    )

    print("initial state")
    pprint.pprint(sim)

    # pairer = UniformPairer(rng)
    pairer = NetworkPairer(N*2, 0.2, rng)
    print(pairer.pretty_print(agents))
    sim.run(pairer)

    print(f"got result")
    pprint.pprint(sim)

if __name__ == "__main__":
    main()
