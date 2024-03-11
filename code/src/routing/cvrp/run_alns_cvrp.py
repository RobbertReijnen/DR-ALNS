import time
from pathlib import Path

from ALNS_custom import ALNS
import autofit_weights

import numpy.random as rnd
import helper_functions

from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
from routing.cvrp.alns_cvrp import cvrp_helper_functions
from routing.cvrp.alns_cvrp import initial_solution, destroy_operators, repair_operators

from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime

PARAMETERS_FILE = './configs/ALNS_cvrp_debug.json'
DEFAULT_RESULTS_ROOT = "./single_runs/"

def run_algo(folder, exp_name, **kwargs):
    print('starting now :-)')
    instance_file = kwargs['instance_file']
    instance_nr = kwargs['instance_nr']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    # LOAD INSTANCE
    base_path = Path(__file__).resolve().parents[0]
    instance_file = str(base_path.joinpath(instance_file))

    nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data = cvrp_helper_functions.read_input_cvrp(instance_file, instance_nr)

    random_state = rnd.RandomState(seed)
    state = cvrpEnv([], nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data, instance_file, seed)
    init_solution = initial_solution.compute_initial_solution(state, random_state)
    print("init_solution: ", init_solution.objective())

    # ALNS
    alns = ALNS(random_state)
    alns.add_destroy_operator(destroy_operators.random_removal)
    alns.add_destroy_operator(destroy_operators.relatedness_removal)
    alns.add_destroy_operator(destroy_operators.neighbor_graph_removal)

    alns.add_repair_operator(repair_operators.regret_insertion)

    weigts = [kwargs["w1"], kwargs["w2"], kwargs['w3'], 0]
    select = RouletteWheel(weigts, decay=kwargs['decay'], num_destroy=3, num_repair=1)
    #accept = SimulatedAnnealing(1, .25, 1 / 100)  # HillClimbing()
    accept = autofit_weights.autofit(SimulatedAnnealing, init_obj=init_solution.objective(), worse=0.05, accept_prob=0.5, num_iters=kwargs['iterations'])
    stop = MaxIterations(iterations)

    # START EVALUATION ALNS
    start_time = time.time()
    if kwargs['degree_of_destruction'] != None:
        nr_nodes_to_remove = round(kwargs['degree_of_destruction'] * nb_customers)
    else:
        nr_nodes_to_remove = None

    result = alns.iterate(init_solution, select, accept, stop, nr_nodes_to_remove=nr_nodes_to_remove)

    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')

    solution = result.best_state
    best_objective = solution.objective()
    print("best_obj", best_objective)
    print(solution.routes)

    helper_functions.write_output(folder, exp_name, kwargs['instance_nr'], kwargs['rseed'], kwargs['iterations'], solution.routes, best_objective, kwargs['instance_file'])

def main(param_file=PARAMETERS_FILE):
    parameters = helper_functions.readJSONFile(param_file)
    folder = DEFAULT_RESULTS_ROOT

    exp_name = str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])
    run_algo(folder, exp_name, **parameters)


if __name__ == "__main__":
    main()