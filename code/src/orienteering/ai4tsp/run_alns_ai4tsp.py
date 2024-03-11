import os
import time
import numpy as np
from pathlib import Path
from env_rl import EnvRL
import autofit_weights

from orienteering.ai4tsp.alns_ai4tsp import initial_solution, repair_operators, destroy_operators
from orienteering.ai4tsp.alns_ai4tsp import ai4tsp_helper_functions

from ALNS_custom import ALNS
from alns.accept import SimulatedAnnealing, HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from orienteering.ai4tsp.alns_ai4tsp.ai4tsp_env import ai4tspEnv

PARAMETERS_FILE = "./configs/ALNS_ai4tsp_debug.json"
DEFAULT_RESULTS_ROOT = "single_runs/"


def run_algo(folder, exp_name, **kwargs):
    print('starting now :-)')
    problem_instance = kwargs['problem_instance']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # LOAD INSTANCE
    base_path = Path(__file__).resolve().parents[1]
    test_data_instance_path = base_path.joinpath('ai4tsp/data/test/instances')
    test_data_adj_path = base_path.joinpath('ai4tsp/data/test/adjs')
    x_path = os.path.join(test_data_instance_path, problem_instance + '.csv')
    adj_path = os.path.join(test_data_adj_path, 'adj-' + problem_instance + '.csv')
    x, adj, problem_instance = ai4tsp_helper_functions.read_instance(x_path, adj_path)
    nodes = [(i + 1) for i in range(0, len(x))]

    random_state = np.random.default_rng(seed)
    state = ai4tspEnv(nodes, [], x, adj, problem_instance, seed)
    init_solution = initial_solution.empty_route(state, init_node=1)

    # ALNS
    alns = ALNS(random_state)
    alns.add_destroy_operator(destroy_operators.random_removal)
    alns.add_destroy_operator(destroy_operators.relatedness_removal)
    alns.add_destroy_operator(destroy_operators.neighbor_graph_removal)

    alns.add_repair_operator(repair_operators.random_best_distance_repair)
    alns.add_repair_operator(repair_operators.random_best_prize_repair)
    alns.add_repair_operator(repair_operators.random_best_ratio_repair)

    pool = None
    weights = [kwargs['w1'], kwargs['w2'], kwargs['w3'], 0]
    select = RouletteWheel(weights, decay=kwargs['decay'], num_destroy=3, num_repair=3)
    init_solution = repair_operators.random_best_prize_repair(init_solution, 0)
    accept = autofit_weights.autofit(SimulatedAnnealing, init_obj=init_solution.objective(), worse=0.05,
                                    accept_prob=0.5, num_iters=kwargs['iterations'])
    stop = MaxIterations(iterations)

    # START EVALUATION SOLUTION (according to the competition rules)
    start_time = time.time()
    result = alns.iterate(init_solution, select, accept, stop, degree_of_destruction=kwargs['dod'], pool=pool)

    solution = result.best_state
    best_objective = - solution.objective()
    elapsed_time = time.time() - start_time

    seed = 19120623
    env = EnvRL(from_file=True, seed=seed, x_path=x_path, adj_path=adj_path)

    for node in solution.route[1:]:
        env.step(node)
    rewards = env.get_collected_rewards()
    pen = env.get_incurred_penalties()
    feas = env.get_feasibility()
    score = rewards + pen
    print('competition_score', score)
    print(' ')

    # Save outputs of main in files
    ai4tsp_helper_functions.write_output(folder, exp_name, kwargs['problem_instance'], kwargs['rseed'], kwargs['iterations'], solution.route, best_objective)


def main(param_file=PARAMETERS_FILE):
    parameters = ai4tsp_helper_functions.readJSONFile(param_file)

    folder = DEFAULT_RESULTS_ROOT
    exp_name = str(parameters["problem_instance"]) + str("_rseed") + str(parameters["rseed"])

    run_algo(folder, exp_name, **parameters)


if __name__ == "__main__":
    main()

