import numpy as np
import pandas as pd
import pickle
from random import Random
from typing import Callable, List, Tuple
from SALib import ProblemSpec

from main import MutNetSimulation, \
    run_mut_net_sim, get_coupling_exclusion, get_correlation_exclusion_popularity, get_agent_data, get_correlation_coupling_popularity

ATTR_THRESHOLD = 0.9

INPUTS = ['prop','density', 'malleability', 'noise', 'sim_sensitivity', 'T']
OUTPUTS = [
    'single_pair_nr', 
    'any_pair_nr', 
    'coupling_deg', 
    'excluded_nr', 
    'excluded_degree', 
    'excluded_prop_m', 
    'excluded_deg_m', 
    'excluded_prop_f', 
    'excluded_deg_f',
    'conv', 
    'pop_corr_coup', 
    'pop_corr_excl']
# sim = MutNetSimulation(
#         num_m=N_m,
#         num_f=N_f,
#         density=density,
#         malleability=malleability,
#         rng=rng,
#         noise=noise,
#         sim_sensitivity=sim_sensitivity,
#         graph_type=graph_type,  # "uniform" or "barabasi"
#         attr_max=10,
#     )

# def get_initial_convergence(sim) -> List:
#     out = []
#     data = get_agent_data(sim, ATTR_THRESHOLD)
#     for t in data:
#         out.append(t[3])
#     return out


def run_model(
        num_m,
        num_f,
        density,
        malleability,
        noise,
        sim_sensitivity,
        graph_type,  # "uniform" or "barabasi"
        T,
        seed = 166):
    
    sim = MutNetSimulation(
        num_m=num_m,
        num_f=num_f,
        density=density,
        malleability=malleability,
        rng = Random(seed),
        noise=noise,
        sim_sensitivity=sim_sensitivity,
        graph_type=graph_type,  # "uniform" or "barabasi"
        attr_max=10
    )
    # get initial data
    initial_agent_data = get_agent_data(sim, ATTR_THRESHOLD)
    clean_initial_data = get_rid_of_exclusion_for_couples(initial_agent_data)

    # run simulation
    run_mut_net_sim(sim, T)

    # get final data
    final_agent_data = get_agent_data(sim, ATTR_THRESHOLD)
    clean_final_data = get_rid_of_exclusion_for_couples(final_agent_data)

    # build the dataframe
    df = turn_tuple_data_into_df(clean_final_data, clean_initial_data, num_m, num_f)

    # get metrics
    metrics = np.array(get_metrics(df))

    # get the popularity correlations
    pop_excl_corr = get_correlation_exclusion_popularity(final_agent_data)[0][1]
    pop_coup_corr = get_correlation_coupling_popularity(final_agent_data)[0][1]

    corr_array = np.array([pop_excl_corr, pop_coup_corr])

    return np.concatenate([metrics, corr_array])

def get_rid_of_exclusion_for_couples(data) -> List[Tuple]:
    out = []
    for tup in data:
        if tup[0] > 0:
            out.append((tup[0], 0, tup[2], tup[3]))
        else:
            out.append(tup)
    return out


def turn_tuple_data_into_df(data, initial_data, num_m, num_f) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=['couple_deg', 'excl_deg', 'pop', 'conv'])
    df_initial = pd.DataFrame(initial_data, columns=['couple_deg_init', 'excl_deg_init', 'pop_init', 'conv_init'])
    df_full = pd.concat([df, df_initial], axis = 1)
    gender = ["m"]*num_m + ['f']*num_f
    df_full['gender'] = gender

    return df_full

def get_metrics(df):
    """
    Gets the metrics of interest.
    """
    single_pair_nr = sum(df['couple_deg'] == 1)
    any_pair_nr = sum(df['couple_deg'] > 0)
    coupling_deg = df[df['couple_deg'] > 0]['couple_deg'].mean()

    excluded_nr = sum(df['excl_deg'] > 0)
    excluded_degree = df[df['excl_deg'] > 0]['excl_deg'].mean()

    # split male and female
    df_m = df[df['gender'] == 'm']
    df_f = df[df['gender'] == 'f']

    excluded_prop_m = sum(df_m['excl_deg'] > 0)/len(df_m['excl_deg'])
    excluded_deg_m = df_m[df_m['excl_deg'] > 0]['excl_deg'].mean()
    excluded_prop_f = sum(df_f['excl_deg'] > 0)/len(df_f['excl_deg'])
    excluded_deg_f = df_f[df_f['excl_deg'] > 0]['excl_deg'].mean()


    conv = (df['conv'] - df['conv_init']).mean()

    return single_pair_nr, any_pair_nr, coupling_deg, excluded_nr, excluded_degree, excluded_prop_m, excluded_deg_m, excluded_prop_f, excluded_deg_f, conv

def model_wrapper_barabasi(X: np.array, func: Callable = run_model):
    """
    :param X: a numpy array in the form: 
        [prop, density, malleability, noise, sim_sensitivity, T] 
    :param func: a function calling the model and returning results
    """
    N, D = X.shape
    results = np.empty((N, len(OUTPUTS)))
    for i in range(N):
        prop,  density, malleability, noise, sim_sensitivity, T = X[i, :]

        num_f = 50
        num_m = int(np.floor(prop*num_f))
        T = int(np.floor(T))
        results[i] = func(num_m, num_f, density, malleability, noise, sim_sensitivity, "barabasi", T)
        
    return results

def model_wrapper_uniform(X: np.array, func: Callable):
    """
    :param X: a numpy array in the form: 
        [prop, density, malleability, noise, sim_sensitivity, T] 
    :param func: a function calling the model and returning results
    """
    prop,  density, malleability, noise, sim_sensitivity, T = X.T
    num_f = 50
    num_m = prop*num_f
    
    return func(num_m, num_f, density, malleability, noise, sim_sensitivity, "uniform", T)
    

def model_wrapper_fully_connected(X: np.array, func: Callable = run_model):
    """
    :param X: a numpy array in the form: 
        [prop,  malleability, noise, sim_sensitivity, T] 
    :param func: a function calling the model and returning results
    """
    prop,  density, malleability, noise, sim_sensitivity, T = X.T
    num_f = 50
    num_m = prop*num_f
    
    return func(num_m, num_f, 1,  malleability, noise, sim_sensitivity, "uniform", T)

def perform_main_sa(sobol_run_param, graph_type):
    sp = ProblemSpec({
        'names': INPUTS,
        'bounds': [
            [1, 4],  # prop
            [0.05,1], # dens
            [0.2,0.6], # mall
            [0, 0.05],  # noise
            [0.01, 1], # sim sensitivity
            [20, 150] # T (is floored later!)
        ],
        'outputs': OUTPUTS
    })
    if graph_type == 'barabasi':
        sp.sample_sobol(2**sobol_run_param).evaluate(model_wrapper_barabasi).analyze_sobol()
    elif graph_type == 'full':
        sp.sample_sobol(2**sobol_run_param).evaluate(model_wrapper_fully_connected).analyze_sobol()

    with open("sa_new.pkl", "wb") as f:
        pickle.dump(sp, f)

    single_pair_nr, any_pair_nr, coupling_deg, excluded_nr, excluded_degree, excluded_prop_m, excluded_deg_m, excluded_prop_f, excluded_deg_f, conv, pop_corr_coup, pop_corr_excl = sp.to_df()
    out_ls = [single_pair_nr, any_pair_nr, coupling_deg, excluded_nr, excluded_degree, excluded_prop_m, excluded_deg_m, excluded_prop_f, excluded_deg_f, conv, pop_corr_coup, pop_corr_excl]
    for i in range(len(out_ls)):
        dfs = out_ls[i]
        total_Si, first_Si, second_Si = dfs
        total_Si.to_csv(f'results\\{OUTPUTS[i]}_total_Si')
        first_Si.to_csv(f'results\\{OUTPUTS[i]}_first_Si')
        second_Si.to_csv(f'results\\{OUTPUTS[i]}_second_Si')

    return sp

def load_sp(path: str) -> ProblemSpec:
    with open(path, 'rb') as f:
        sp = pickle.load(f)
    return sp
    

def run_purely_randomised_sa(sample_nr: int, graph_type):
    ls = []
    seeds = np.linspace(1, sample_nr, sample_nr).astype(int).tolist()
    N_f = 50
    N_m = 3* N_f
    T = 100
    if graph_type == 'barabasi':
        density = 0.2
    elif graph_type == 'full':
        density = 1
        graph_type = 'uniform'
    noise = 0.01
    malleability = 0.4
    sim_sensitivity = 0.1
    
    for seed in seeds:
        results = run_model(
            num_m=N_m,
            num_f=N_f,
            density=density,
            malleability=malleability,
            noise=noise,
            sim_sensitivity=sim_sensitivity,
            graph_type=graph_type,
            T=T,
            seed=seed
        )
        ls.append(results)

    out = pd.DataFrame(ls, columns=OUTPUTS)
    out.to_csv(f'sa_randomised_{graph_type}.csv')
    return out


def tester():
    
    T = 10
    density = 0.2
    noise = 0.01
    malleability = 0.4
    sim_sensitivity = 0.1
    graph_type = "barabasi"
    # NOTE: indices go from males -> females --- offset females by N_m
    N_m = 50
    N_f = 2 * N_m
    run_model(
        N_m,
        N_f,
        density,
        malleability,
        noise,
        sim_sensitivity,
        graph_type,
        T
    )

run_purely_randomised_sa(100, "barabasi")
sp = perform_main_sa(7, 'barabasi')
# run_purely_randomised_sa(100, "full")
# sp = perform_main_sa(8, 'full')
