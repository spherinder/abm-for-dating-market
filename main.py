from collections.abc import Iterable, Sequence
from math import exp, floor
from random import Random
import numpy as np
import pprint
import datetime

from dataclasses import dataclass, field
from typing import Final, TypeAlias, override
from rustworkx.visualization import mpl_draw
from rustworkx import (
    AllPairsPathMapping,
    PyDiGraph,
    PyGraph,
    all_pairs_dijkstra_shortest_paths,
    circular_layout,
    undirected_gnp_random_graph,
    barabasi_albert_graph,
)

from src.utils import gumbel_weighted_permutation
import matplotlib.pyplot as plt
from matplotlib.cm import Blues, cividis
import matplotlib.animation as animation


Rngs: TypeAlias = tuple[Random, np.random.Generator]


@dataclass
class Agent:
    # Mostly change this
    attr: Final[float]
    sought: float
    rng: Random = field(repr=False)

    @classmethod
    def new(cls, attr_max: int, rng: Random):
        return cls(rng.randint(0, attr_max), rng.randint(0, attr_max), rng)


def pair_up(
    males: Sequence[int], fems: Sequence[int], paths: AllPairsPathMapping, rngs: Rngs
) -> Iterable[tuple[int, int]]:
    num_m = len(males)
    num_f = len(fems)
    maleixs = list(range(num_m))
    femixs = list(range(num_f))

    if num_m > num_f:
        maleixs = rngs[0].sample(maleixs, num_f)
    elif num_m < num_f:
        femixs = rngs[0].sample(femixs, num_m)
    pathmaps = (paths[males[mi]] for mi in maleixs)
    weights = np.array(
        [
            [(len(pathmap[fems[fi]]) if fems[fi] in pathmap else 0) for fi in femixs]
            for pathmap in pathmaps
        ]
    )

    perm = gumbel_weighted_permutation(weights, rngs[1])
    return zip(maleixs, (femixs[int(i)] for i in perm))


class MutNetSimulation:
    graph: PyGraph[int]
    density: Final[float]
    males: Final[list[Agent]]
    fems: Final[list[Agent]]
    attr_max: Final[int]
    malleability: Final[float]  # ∈ [0,1]
    rngs: Rngs
    noise: float
    sim_sensitivity: float

    def __init__(
        self,
        num_m: int,
        num_f: int,
        density: float,
        malleability: float,
        rng: Random,
        noise: float,
        sim_sensitivity: float,
        graph_type: str,
        attr_max: int = 50,
    ):
        seed = rng.randint(0, 2**32 - 1)
        # Replace this with other graph generators
        # Read: https://www.rustworkx.org/api/random_graph_generator_functions.html
        graphs = {
            "uniform": undirected_gnp_random_graph(num_m + num_f, density, seed),
            "barabasi": barabasi_albert_graph(
                num_m + num_f, floor((num_m + num_f) * density), seed
            ),
        }
        self.graph = graphs[graph_type]

        self.males = [Agent.new(attr_max, rng) for _ in range(num_m)]
        self.fems = [Agent.new(attr_max, rng) for _ in range(num_f)]

        self.rngs = (rng, np.random.default_rng(seed))
        self.density = density
        self.malleability = malleability
        self.noise = noise
        self.sim_sensitivity = sim_sensitivity
        self.attr_max = attr_max

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        # return (self.attr_max - abs(a_i.sought - a_j.attr)) / self.attr_max
        return exp(-self.sim_sensitivity * abs(a_i.sought - a_j.attr))

    def couple(self, mi: int, fi: int):
        # if not self.graph.has_edge(mi, len(self.males) + fi):
        #     _ = self.graph.add_edge(mi, len(self.males) + fi, None)
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

    def step_local(self, log: bool = False):
        paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        num_m = len(self.males)
        pairs = pair_up(
            range(num_m), range(num_m, num_m + len(self.fems)), paths, self.rngs
        )
        pairs = list(pairs)
        for mi, fi in pairs:
            a_m = self.males[mi]
            a_f = self.fems[fi]
            m_accepts = self.rngs[0].random() < self.accept_prob(a_m, a_f)
            f_accepts = self.rngs[0].random() < self.accept_prob(a_f, a_m)
            if log:
                print(
                    f"[{m_accepts} {mi} {fi} {f_accepts}]: {self.males[mi]} | {self.fems[fi]}"
                )
            if m_accepts and f_accepts:
                self.couple(mi, fi)
                if log:
                    print(
                        f"coupled {self.males[mi]} | {self.fems[fi]}, nedges {len(self.graph.edge_list())}"
                    )
            # elif self.graph.has_edge(mi, len(self.males)+fi):
            #     self.graph.remove_edge(mi, len(self.males)+fi)
            if not m_accepts:
                self.rejects(a_m, a_f)
            if not f_accepts:
                self.rejects(a_f, a_m)

    # do this for only 1 pair -> need to do N times more to balance out
    def step_filter(self, num_attr: int):
        pass

    def create_noise(self):
        rng = self.rngs[0]
        # Adding edges
        if self.rngs[0].random() < self.noise:
            a = rng.randint(0, len(self.males) + len(self.fems) - 1)
            b = rng.randint(0, len(self.males) + len(self.fems) - 1)
            if a != b and not self.graph.has_edge(a, b):
                self.graph.add_edge(a, b, None)

        # Deleting edges
        if self.rngs[0].random() < self.noise:
            a = rng.randint(0, len(self.males) + len(self.fems))
            b = rng.randint(0, len(self.males) + len(self.fems))
            if a != b and self.graph.has_edge(a, b):
                self.graph.remove_edge(a, b)

    @override
    def __repr__(self) -> str:
        males = "\n    ".join(f"{a}" for a in self.males)
        fems = "\n    ".join(f"{a}" for a in self.fems)
        return (
            f"MutNetSimulation(attr_max={self.attr_max}, malleability={self.malleability},"
            f"\n  males=[\n    {males}\n  ],\n  females=[\n    {fems}\n  ]\n)"
        )


def format_graph_edges(g: PyGraph, n: int):
    return ", ".join(
        f"({"F"+str(v1-n) if v1 > n else "M"+str(v1)},{"F"+str(v2-n) if v2 > n else "M"+str(v2)})"
        for v1, v2 in g.edge_list()
    )


def run_mut_net_sim_viz():
    # Defining initial values
    SEED = 42
    rng = Random(SEED)
    T = 200
    density = 0.2
    noise = 0.1
    malleability = 0.1
    sim_sensitivity = 0.5
    graph_type = "barabasi"
    N = 4

    # Creating social graph (underlying structure)
    sim = MutNetSimulation(
        num_m=N,
        num_f=2 * N,
        density=density,
        malleability=malleability,
        rng=rng,
        noise=noise,
        sim_sensitivity=sim_sensitivity,
        graph_type=graph_type,  # "uniform" or "barabasi"
        attr_max=10,
    )
    pos = circular_layout(sim.graph)

    pprint.pprint(sim)

    # Creating attraction graph
    attraction_graph: PyDiGraph[None, float] = PyDiGraph()
    _ = attraction_graph.add_nodes_from(None for _ in range(N + N))
    _ = attraction_graph.add_edges_from(
        e
        for mi in range(N)
        for fi in range(N)
        for e in [(mi, fi + N, 0), (fi + N, mi, 0)]
    )
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int):
        ax.clear()

        print("step frame", frame)
        sim.create_noise()
        sim.step_local()

        # edge_widths = [data['weight'] for data in sim.graph.edges()]

        mpl_draw(
            sim.graph,
            pos,
            ax=ax,
            with_labels=True,
            alpha=0.6,
            # width=edge_widths,
            # edge_color=edge_widths,
            edge_cmap=Blues,
            edge_vmin=0,
            edge_vmax=3,
        )

        for mi in range(N):
            for fi in range(N):
                attraction_graph.update_edge(
                    mi, fi + N, sim.accept_prob(sim.males[mi], sim.fems[fi])
                )
                attraction_graph.update_edge(
                    fi + N, mi, sim.accept_prob(sim.fems[fi], sim.males[mi])
                )

        colors = ["skyblue" for _ in range(N)]
        colors.extend("pink" for _ in range(N))

        # print("edges", attraction_graph.edges())

        mpl_draw(
            attraction_graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color=colors,
            node_size=500,
            font_color="black",
            width=[2 * p**6 for p in attraction_graph.edges()],
            # arrow_size=[int(10*p) for p in attraction_graph.edges()],
            edge_color="red",
            edge_cmap=cividis,
        )

        _ = ax.set_title(
            f"Time Step: {frame}"
        )  # pyright: ignore[reportUnknownMemberType]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=400)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    ani.save(
        f"asset/graph_{now}.mp4", writer="ffmpeg"
    )  # pyright: ignore[reportUnknownMemberType]
    print(f"Saved mp4 {graph_type}_graph_{now}.mp4 successfully")


def main():
    # run_static_net_sim()
    # run_mut_net_sim()
    run_mut_net_sim_viz()


if __name__ == "__main__":
    main()
