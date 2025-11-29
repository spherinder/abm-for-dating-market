from collections.abc import Iterable, Sequence
from math import exp, floor
from random import Random
import numpy as np
import pprint
import datetime

from dataclasses import dataclass, field
from typing import Final, Literal, Self, TypeAlias, TypeVar, override
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
T = TypeVar("T", bound=np.generic)
Vec: TypeAlias = np.ndarray[tuple[int], np.dtype[T]]


@dataclass
class Agent:
    attr: Final[Vec[np.integer]]
    sought: Vec[np.floating]
    rng: np.random.Generator = field(repr=False)  # (don't print)

    @classmethod
    def new(cls, attr_dim: int, attr_max: int, rng: np.random.Generator) -> Self:
        return cls(
            rng.integers(0, attr_max, (attr_dim,)),
            rng.uniform(0, attr_max, (attr_dim,)),
            rng,
        )


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
    attr_dim: Final[int]
    graph_type: Literal["barabasi", "uniform"]

    # Data collecting part
    avg_coupling: list[float]
    avg_exclusion: list[float]
    convergence: list[float]  # mean over decision values (accept_prob)

    def __init__(
        self,
        num_m: int,
        num_f: int,
        density: float,
        malleability: float,
        rng: Random,
        noise: float,
        sim_sensitivity: float,
        graph_type: Literal["barabasi", "uniform"] = "barabasi",
        attr_max: int = 50,
        attr_dim: int = 2,
    ):
        seed = rng.randint(0, 2**32 - 1)
        self.rngs = (rng, np.random.default_rng(seed))
        self.graph_type = graph_type
        # Replace this with other graph generators
        # Read: https://www.rustworkx.org/api/random_graph_generator_functions.html
        if graph_type == "uniform":
            self.graph = undirected_gnp_random_graph(num_m + num_f, density, seed)
        elif graph_type == "barabasi":
            self.graph = barabasi_albert_graph(
                num_m + num_f, floor((num_m + num_f) * density), seed
            )

        self.males = [Agent.new(attr_dim, attr_max, self.rngs[1]) for _ in range(num_m)]
        self.fems = [Agent.new(attr_dim, attr_max, self.rngs[1]) for _ in range(num_f)]
        self.density = density
        self.malleability = malleability
        self.noise = noise
        self.sim_sensitivity = sim_sensitivity
        self.attr_max = attr_max
        self.attr_dim = attr_dim

        self.avg_coupling = []
        self.avg_exclusion = []
        self.convergence = []

    def accept_prob(self, a_i: Agent, a_j: Agent) -> float:
        return exp(-self.sim_sensitivity * np.linalg.norm(a_i.sought - a_j.attr))

    def couple(self, mi: int, fi: int):
        # if not self.graph.has_edge(mi, len(self.males) + fi):
        #     _ = self.graph.add_edge(mi, len(self.males) + fi, None)
        m = self.males[mi]
        f = self.fems[fi]
        self.approach(m, f)
        self.approach(f, m)

    def approach(self, a_i: Agent, a_j: Agent):
        diff = a_j.attr - a_i.sought
        norm2 = np.inner(diff, diff)  # pyright: ignore[reportAny]
        a_i.sought += diff * min(
            1.0, self.malleability / (norm2 + 0.01)
        )  # pyright: ignore[reportAny]

    def rejects(self, a_i: Agent, a_j: Agent):
        pass
        # diff = a_i.attr - a_j.sought
        # norm2 = np.inner(diff,diff)  # pyright: ignore[reportAny]
        # a_j.sought = np.clip(
        #     a_j.sought + self.malleability * (diff / (norm2 + 0.01)),  # pyright: ignore[reportAny]
        #     0, self.attr_max
        # )

    def step_local(self, log: bool = False):
        paths = all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        num_m = len(self.males)
        num_f = len(self.fems)
        num_a = num_m + num_f
        pairs = pair_up(range(num_m), range(num_m, num_m + num_f), paths, self.rngs)
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

        # Collecting data for this round
        t = 0.5  # change this as you seem fit -> might wanna define this globally somewhere
        avg_coupling = 0
        avg_exclusion = 0
        convergence = 0

        # res = (coupling, exclusion, pop, convergence)
        for a in range(num_a):
            res = get_coupling_exclusion(self, a, t)
            avg_coupling += res[0]
            avg_exclusion += res[1]
            convergence += res[3]
        avg_coupling /= num_a
        avg_exclusion /= num_a
        convergence /= num_a
        self.avg_coupling.append(avg_coupling)
        self.avg_exclusion.append(avg_exclusion)
        self.convergence.append(convergence)

    # do this for only 1 pair -> need to do N times more to balance out
    def step_filter(self, num_attr: int):
        pass

    def create_noise(self):
        rng = self.rngs[0]
        # Adding edges
        if self.rngs[0].random() < self.noise:
            while True:
                a = rng.randint(0, len(self.males) + len(self.fems) - 1)
                b = rng.randint(0, len(self.males) + len(self.fems) - 1)
                if a != b and not self.graph.has_edge(a, b):
                    _ = self.graph.add_edge(a, b, None)
                    break

        # Deleting edges
        if self.rngs[0].random() < self.noise:
            while True:
                a = rng.randint(0, len(self.males) + len(self.fems) - 1)
                b = rng.randint(0, len(self.males) + len(self.fems) - 1)
                if a != b and self.graph.has_edge(a, b):
                    self.graph.remove_edge(a, b)
                    break

    # @override
    # def __repr__(self) -> str:
    #     males = "\n    ".join(f"{a}" for a in self.males)
    #     fems = "\n    ".join(f"{a}" for a in self.fems)
    #     return (
    #         f"{self.__class__.__name__}(attr_max={self.attr_max}, malleability={self.malleability},"
    #         f"\n  males=[\n    {males}\n  ],\n  females=[\n    {fems}\n  ]\n)"
    #     )


def format_graph_edges(g: PyGraph, n: int):
    return ", ".join(
        f"({"F"+str(v1-n) if v1 > n else "M"+str(v1)},{"F"+str(v2-n) if v2 > n else "M"+str(v2)})"
        for v1, v2 in g.edge_list()
    )


# Return a tuple of (coupling, exclusion, pop, convergence)
def get_coupling_exclusion(
    sim: MutNetSimulation, a: int, t: float
) -> tuple[int, int, int, float]:
    neighbors = sim.graph.neighbors(a)
    deg = [0, 0, len(neighbors), 0.0]
    for b in neighbors:
        # check if same genders
        if (a < len(sim.males) and b < len(sim.males)) or (
            a >= len(sim.males) and b >= len(sim.males)
        ):
            continue

        # map indices to Agent classes
        if a < len(sim.males):
            agent_a = sim.males[a]
            agent_b = sim.fems[b - len(sim.males)]
        else:
            agent_a = sim.fems[a - len(sim.males)]
            agent_b = sim.males[b]

        # Q(a, b) to be used to calculate "convergence"
        deg[3] += sim.accept_prob(agent_a, agent_b)

        # deg of coupling
        if (
            sim.accept_prob(agent_a, agent_b) > t
            and sim.accept_prob(agent_b, agent_a) > t
        ):
            deg[0] += 1

        # deg of exclusion
        if (
            sim.accept_prob(agent_a, agent_b) > t
            and sim.accept_prob(agent_b, agent_a) <= t
        ):
            deg[1] += 1

    return tuple(deg)


# Use this to get data of all agents to get info *AT that round*
# Return list of (coupling, exclusion, pop, convergence)
def get_agent_data(sim: MutNetSimulation, t: float) -> list[tuple[int, int, int]]:
    data = []
    for a in range(len(sim.males) + len(sim.fems)):
        data.append(get_coupling_exclusion(sim, a, t))
    return data


def get_correlation_exclusion_popularity(
    data: list[tuple[int, int, int]],
) -> np.ndarray:
    data_np = np.array(data)
    exclusion_pop_array = data_np[:, [1, 2]]

    return np.corrcoef(exclusion_pop_array, rowvar=False)

def get_correlation_coupling_popularity(
    data: list[tuple[int, int, int]],
) -> np.ndarray:
    data_np = np.array(data)
    exclusion_pop_array = data_np[:, [0, 2]]

    return np.corrcoef(exclusion_pop_array, rowvar=False)


def run_mut_net_sim_viz(sim: MutNetSimulation, T: int, live_view=False):
    if live_view:
        # from matplotlib import use

        # use("TkAgg")
        pass
    N_m = len(sim.males)
    N_f = len(sim.fems)
    pos = circular_layout(sim.graph)

    pprint.pprint(sim)

    # Creating attraction graph
    attraction_graph: PyDiGraph[tuple[Vec[np.integer], Vec[np.floating]], float] = (
        PyDiGraph()
    )
    _ = attraction_graph.add_nodes_from(
        (sim.males[i].attr, sim.males[i].sought) for i in range(N_m)
    )
    _ = attraction_graph.add_nodes_from(
        (sim.fems[i].attr, sim.fems[i].sought) for i in range(N_f)
    )
    _ = attraction_graph.add_edges_from(
        e
        for mi in range(N_m)
        for fi in range(N_f)
        for e in [
            (mi, fi + N_m, sim.accept_prob(sim.males[mi], sim.fems[fi])),
            (fi + N_m, mi, sim.accept_prob(sim.fems[fi], sim.males[mi])),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int):
        ax.clear()

        print("step frame", frame)
        sim.create_noise()
        sim.step_local()

        _ = mpl_draw(
            sim.graph,
            pos,
            ax=ax,
            # with_labels=True,
            alpha=0.6,
            # width=edge_widths,
            # edge_color=edge_widths,
            edge_cmap=Blues,
            edge_vmin=0,
            edge_vmax=3,
        )

        for i in range(N_m):
            attraction_graph[i] = (sim.males[i].attr, sim.males[i].sought)
        for i in range(N_f):
            attraction_graph[i + N_m] = (sim.fems[i].attr, sim.fems[i].sought)
        for mi in range(N_m):
            for fi in range(N_f):
                attraction_graph.update_edge(
                    mi, fi + N_m, sim.accept_prob(sim.males[mi], sim.fems[fi])
                )
                attraction_graph.update_edge(
                    fi + N_m, mi, sim.accept_prob(sim.fems[fi], sim.males[mi])
                )

        colors = ["skyblue" for _ in range(N_m)]
        colors.extend("pink" for _ in range(N_f))

        # print("edges", attraction_graph.edges())

        mpl_draw(
            attraction_graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=lambda node: f"a:{node[0]}\ns:{np.array2string(node[1], precision=1)}",
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

    ani = animation.FuncAnimation(fig, update, frames=T, interval=400, repeat=False)
    if live_view:
        plt.show()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    ani.save(
        f"asset/{sim.graph_type}_graph_{now}.mp4", writer="ffmpeg"
    )  # pyright: ignore[reportUnknownMemberType]
    print(f"Saved mp4 {sim.graph_type}_graph_{now}.mp4 successfully")


def run_mut_net_sim(sim: MutNetSimulation, T: int):
    pprint.pprint(sim)

    for _ in range(T):
        sim.create_noise()
        sim.step_local()
    return sim


def main():
    SEED = 166
    rng = Random(SEED)
    T = 50
    density = 0.2
    noise = 0.01
    malleability = 0.4
    sim_sensitivity = 0.1
    graph_type = "barabasi"
    # NOTE: indices go from males -> females --- offset females by N_m
    N_m = 4
    N_f = 2 * N_m

    # Creating social graph (underlying structure)
    sim = MutNetSimulation(
        num_m=N_m,
        num_f=N_f,
        density=density,
        malleability=malleability,
        rng=rng,
        noise=noise,
        sim_sensitivity=sim_sensitivity,
        graph_type=graph_type,  # "uniform" or "barabasi"
        attr_max=10,
    )
    # run_mut_net_sim_viz(sim, T, live_view=False)
    run_mut_net_sim(sim, T)
    print("Males attributes and soughts")
    for i in range(len(sim.males)):
        print(f"{i}: {sim.males[i]}")
    print("Females attributes and soughts")
    for i in range(len(sim.fems)):
        print(f"{i}: {sim.fems[i]}")
    """
        Data Collection Examples

        print("deg of coupling & exclusion and convergence")
        print(f"{sim.avg_coupling}")
        print(f"{sim.avg_exclusion}")
        print(f"{sim.convergence}")

        print("final agent data")
        print(get_agent_data(sim, 0.5))

        print("correlation")
        data = get_agent_data(sim, 0.5)
        corr_matrix = get_correlation_exclusion_popularity(data)
        print(f"corr: {corr_matrix[0, 1]}")
    """


if __name__ == "__main__":
    main()
