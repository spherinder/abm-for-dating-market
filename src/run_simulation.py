# src/run_simulation.py

from .agents.simple import SimpleAgent
from .matchmaking.random_uniform import RandomUniformMatching
from .models.simulation import Simulation


def main():
    agent_num_male = 5
    agent_num_female = 5

    # --- 1) CREATE AGENTS ---
    agents = []
    agent_id = 0
    for i in range(agent_num_male):
        agents.append(SimpleAgent(id=agent_id, a=i, u=5, gender="M"))
        agent_id += 1
    for i in range(agent_num_female):
        agents.append(SimpleAgent(id=agent_id, a=i, u=5, gender="F"))
        agent_id += 1

    # --- 2) PICK MATCHMAKING MODEL ---
    matchmaking_model = RandomUniformMatching()

    # --- 3) BUILD SIMULATION ENGINE ---
    sim = Simulation(
        agents=agents,
        matchmaking_model=matchmaking_model,
        T=20
    )

    # --- 4) RUN IT ---
    D = sim.run()

    # --- 5) DEMO OUTPUT ---
    print("Final dating matrix D:")
    print(D)


if __name__ == "__main__":
    main()
