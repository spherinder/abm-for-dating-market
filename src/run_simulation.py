# src/run_simulation.py

import random

import numpy as np
from .agents.simple import SimpleAgent
from .matchmaking.random_uniform import RandomUniformMatching
from .models.simulation import Simulation


def main():
    agent_num_male = 50
    agent_num_female = 50

    # --- 1) CREATE AGENTS ---
    agents = []
    agent_id = 0
    for i in range(agent_num_male):
        agent_a = random.randint(0, 50)
        agent_u = random.randint(0, 50)
        agent_male = SimpleAgent(id=agent_id, a=agent_a, u=agent_u, gender="M")
        agents.append(agent_male)
        agent_id += 1
    for i in range(agent_num_female):
        agent_a = random.randint(0, 50)
        agent_u = random.randint(0, 50)
        agent_female = SimpleAgent(id=agent_id, a=agent_a, u=agent_u, gender="F")
        agents.append(agent_female)
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
    print(f"Out of {len(agents)} agents, {np.sum(D)/2} were matched.")


if __name__ == "__main__":
    main()
