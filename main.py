from collections.abc import Iterable, Sequence
from math import exp
from random import Random, uniform
import numpy as np
import pprint

from dataclasses import dataclass, field
from typing import Final, TypeAlias, override
from rustworkx.generators import complete_graph, directed_complete_graph
from rustworkx.visualization import mpl_draw
from sortedcontainers import SortedSet
from rustworkx import AllPairsPathMapping, PyDiGraph, PyGraph, all_pairs_dijkstra_shortest_paths, circular_layout, spring_layout, undirected_gnp_random_graph

from src.utils import gumbel_weighted_permutation
import matplotlib.pyplot as plt
from matplotlib.cm import Blues, cividis
import matplotlib.animation as animation


Rngs: TypeAlias = tuple[Random,np.random.Generator]

@dataclass
class Agent:
    attr: Final[float]
    sought: float
    rng: Random = field(repr=False)

    @classmethod
    def new(cls, attr_max: int, rng: Random):
        return cls(rng.randint(0, attr_max), rng.randint(0, attr_max), rng)

@dataclass
class UniformPairer:
    rng: Random = field(repr=False)

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

def pair_up(males: Sequence[int], fems: Sequence[int], paths: AllPairsPathMapping, rngs: Rngs) -> Iterable[tuple[int,int]]:
    num_m = len(males)
    num_f = len(fems)
    maleixs = list(range(num_m))
    femixs = list(range(num_f))

    if num_m > num_f:
        maleixs = rngs[0].sample(maleixs, num_f)
    elif num_m < num_f:
        femixs = rngs[0].sample(femixs, num_m)
    pathmaps = (paths[males[mi]] for mi in maleixs)
    weights = np.array([
        [ (len(pathmap[fems[fi]]) if fems[fi] in pathmap else 0) for fi in femixs ]
        for pathmap in pathmaps
    ])

    perm = gumbel_weighted_permutation(weights, rngs[1])
    return zip(maleixs, (femixs[int(i)] for i in perm))


class NetworkPairer:
    paths: Final[AllPairsPathMapping]
    rngs: Rngs
    graph: Final[PyGraph[int]] # not necessary to store this
    density: Final[float] # not necessary to store this

    def __init__(self, n_agents: int, density: float, rngs: Rngs):
        seed = rngs[0].randint(0, 2**32 - 1)
        self.graph = undirected_gnp_random_graph(n_agents, density, seed)
        self.paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        self.rngs = rngs
        self.density = density

    def pair_up(self, males: Sequence[int], fems: Sequence[int]) -> Iterable[tuple[int,int]]:
        return pair_up(males, fems, self.paths, self.rngs)

    def pretty_print(self, agents: list[Agent]) -> str:
        edges = "\n  ".join(
            ", ".join(f"{(agents[edge[0]], agents[edge[1]])}" for edge in self.graph.edge_list()[i:i+2])
            for i in range(0, len(self.graph.edge_list()), 5))
        return f"NetworkPairer(n_agents={len(self.graph)}, density={self.density}, Edges=[\n  {edges}\n])"


class Simulation:
    agents: Final[list[Agent]]
    males: SortedSet[int]
    fems: SortedSet[int]
    couples: list[tuple[int,int]]
    attr_max: Final[int]
    rounds: Final[int]
    malleability: Final[float] # ∈ [0,1]
    rngs: Rngs

    def __init__(
        self, males: Iterable[Agent], fems: Iterable[Agent],
        rounds: int, malleability: float, rng: Random, attr_max: int = 50
    ):
        seed = rng.randint(0, 2**32 - 1)
        self.rngs = (rng, np.random.default_rng(seed))

        self.agents = list(males)
        num_m = len(self.agents)
        self.agents.extend(fems)
        self.males = SortedSet(range(num_m))
        self.fems = SortedSet(range(num_m, len(self.agents)))

        self.couples = []
        self.malleability = malleability
        self.attr_max = attr_max
        self.rounds = rounds

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        return (self.attr_max - abs(a_i.sought - a_j.attr)) / self.attr_max

    def rejects(self, a_i: Agent, a_j: Agent):
        a_j.sought += self.malleability * (a_j.sought - a_i.attr) # TODO
        if a_j.sought < 0:
            a_j.sought = 0
        if a_j.sought > self.attr_max:
            a_j.sought = self.attr_max

    def couple(self, m: int, f: int):
        self.males.remove(m)
        self.fems.remove(f)
        self.couples.append((m,f))

    def run(self, pairer: UniformPairer | NetworkPairer):
        for _ in range(self.rounds):
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
                m_accepts = self.rngs[0].random() < self.accept_prob(a_m, a_f)
                f_accepts = self.rngs[0].random() < self.accept_prob(a_f, a_m)
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
        return (f"Simulation(attr_max={self.attr_max}, malleability={self.malleability}, rounds={self.rounds},"
           f"\n  males=[\n    {males}\n  ],\n  females=[\n    {fems}\n  ],\n  couples=[\n    {couples}\n  ]\n)"
           )


class MutNetSimulation:
    graph: PyGraph[int]
    density: Final[float]
    males: Final[list[Agent]]
    fems: Final[list[Agent]]
    attr_max: Final[int]
    malleability: Final[float] # ∈ [0,1]
    rngs: Rngs

    def __init__(
        self, num_m: int, num_f: int, density: float,
        malleability: float, rng: Random, attr_max: int = 50
    ):
        seed = rng.randint(0, 2**32 - 1)
        self.graph = undirected_gnp_random_graph(num_m+num_f, density, seed)

        self.males = [Agent.new(attr_max, rng) for _ in range(num_m)]
        self.fems = [Agent.new(attr_max, rng) for _ in range(num_f)]

        self.rngs = (rng, np.random.default_rng(seed))
        self.density = density
        self.malleability = malleability
        self.attr_max = attr_max

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        # return (self.attr_max - abs(a_i.sought - a_j.attr)) / self.attr_max
        return exp(-0.5*abs(a_i.sought - a_j.attr))

    def couple(self, mi: int, fi: int):
        if not self.graph.has_edge(mi, len(self.males) + fi):
            _ = self.graph.add_edge(mi, len(self.males) + fi, None)
        m = self.males[mi]
        f = self.fems[fi]
        m.sought += self.malleability * (1 if m.sought < f.attr else -1)
        f.sought += self.malleability * (1 if f.sought < m.attr else -1)

    def rejects(self, a_i: Agent, a_j: Agent):
        a_j.sought += self.malleability * (1 if a_j.sought > a_i.attr else -1)
        if a_j.sought < 0:
            a_j.sought = 0
        if a_j.sought > self.attr_max:
            a_j.sought = self.attr_max

    def step(self, log: bool = False):
        paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        num_m = len(self.males)
        pairs = pair_up(range(num_m), range(num_m, num_m + len(self.fems)), paths, self.rngs)
        pairs = list(pairs)
        for mi, fi in pairs:
            a_m = self.males[mi]
            a_f = self.fems[fi]
            m_accepts = self.rngs[0].random() < self.accept_prob(a_m, a_f)
            f_accepts = self.rngs[0].random() < self.accept_prob(a_f, a_m)
            if log: print(f"[{m_accepts} {mi} {fi} {f_accepts}]: {self.males[mi]} | {self.fems[fi]}")
            if m_accepts and f_accepts:
                self.couple(mi, fi)
                if log: print(f"coupled {self.males[mi]} | {self.fems[fi]}, nedges {len(self.graph.edge_list())}")
            # elif self.graph.has_edge(mi, len(self.males)+fi):
            #     self.graph.remove_edge(mi, len(self.males)+fi)
            if not m_accepts:
                self.rejects(a_m,a_f)
            if not f_accepts:
                self.rejects(a_f,a_m)


    @override
    def __repr__(self) -> str:
        males = "\n    ".join(f"{a}" for a in self.males)
        fems = "\n    ".join(f"{a}" for a in self.fems)
        return (f"MutNetSimulation(attr_max={self.attr_max}, malleability={self.malleability},"
           f"\n  males=[\n    {males}\n  ],\n  females=[\n    {fems}\n  ]\n)"
           )

def run_static_net_sim():
    SEED = 42
    rng = Random(SEED)

    N = 10 # agents per gender
    males = (Agent.new(50, rng) for _ in range(N))
    fems = (Agent.new(50, rng) for _ in range(N))
    sim = Simulation(males, fems, 20, 0.1, rng)

    print("initial state")
    pprint.pprint(sim)

    # pairer = UniformPairer(rng)
    pairer = NetworkPairer(N*2, 0.2, sim.rngs)
    print(pairer.pretty_print(sim.agents))
    sim.run(pairer)

    print(f"got result")
    pprint.pprint(sim)


def format_graph_edges(g: PyGraph, n: int):
    return ", ".join(
        f"({"F"+str(v1-n) if v1 > n else "M"+str(v1)},{"F"+str(v2-n) if v2 > n else "M"+str(v2)})"
        for v1,v2 in g.edge_list()
    )

def run_mut_net_sim():
    SEED = 42
    T = 100
    rng = Random(SEED)

    N = 3
    sim = MutNetSimulation(N, N, 0.5, 0.1, rng, 10)

    print("initial state")
    pprint.pprint(sim)

    print("graph", format_graph_edges(sim.graph, N))

    for _ in range(T):
        sim.step(log=True)

    print(f"got result")
    pprint.pprint(sim)

    print("graph", format_graph_edges(sim.graph, N))

def run_mut_net_sim_viz():
    SEED = 42
    T = 100
    rng = Random(SEED)

    N = 3
    sim = MutNetSimulation(N, N, 0.5, 0.1, rng, 10)
    pos = circular_layout(sim.graph)

    pprint.pprint(sim)

    attraction_graph: PyDiGraph[None,float] = PyDiGraph()
    _ = attraction_graph.add_nodes_from(None for _ in range(N+N))
    _ = attraction_graph.add_edges_from(e for mi in range(N) for fi in range(N) for e in [(mi,fi+N,0),(fi+N,mi,0)])
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int):
        ax.clear()

        print("step frame", frame)
        sim.step()

        # edge_widths = [data['weight'] for data in sim.graph.edges()]

        mpl_draw(
            sim.graph, pos, ax=ax, with_labels=True, alpha=0.6,
            # width=edge_widths,
            # edge_color=edge_widths,
            edge_cmap=Blues,
            edge_vmin=0, edge_vmax=3
        )

        for mi in range(N):
            for fi in range(N):
                attraction_graph.update_edge(mi, fi+N, sim.accept_prob(sim.males[mi],sim.fems[fi]))
                attraction_graph.update_edge(fi+N, mi, sim.accept_prob(sim.fems[fi],sim.males[mi]))

        colors = ['skyblue' for _ in range(N)]
        colors.extend('pink' for _ in range(N))

        # print("edges", attraction_graph.edges())

        mpl_draw(
            attraction_graph, pos=pos, ax=ax, with_labels=True,
            node_color=colors,
            node_size=500, font_color='black',
            width=[4*p for p in attraction_graph.edges()],
            # arrow_size=[int(10*p) for p in attraction_graph.edges()],
            edge_color='red',
            edge_cmap=cividis,
        )

        _ = ax.set_title(f"Time Step: {frame}")  # pyright: ignore[reportUnknownMemberType]


    ani = animation.FuncAnimation(fig, update, frames=T, interval=400)
    ani.save('dynamic_graph.mp4', writer='ffmpeg')  # pyright: ignore[reportUnknownMemberType]
    print("saved mp4 successfully")

def main():
    # run_static_net_sim()
    # run_mut_net_sim()
    run_mut_net_sim_viz()

if __name__ == "__main__":
    main()
