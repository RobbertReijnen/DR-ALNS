import csv
from pathlib import Path
import time
from rl.environments.ai4tsp_AlnsEnv_LSA1 import ai4tspAlnsEnv_LSA1
import helper_functions

from stable_baselines3 import PPO

DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = "configs/drl_alns_ai4tsp_debug.json"


def run_algo(folder, exp_name, client=None, **kwargs):
    print('starting now :-)')
    instance = kwargs['problem_instance']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / kwargs['model_directory'] / 'model'
    model = PPO.load(model_path)

    parameters = {'environment': {'iterations': iterations, 'instances': [instance]}}
    env = ai4tspAlnsEnv_LSA1(parameters)
    start_time = time.time()
    env.run(model)
    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')

    best_objective = -env.best_solution.objective()

    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective'])
        writer.writerow([instance, seed, iterations, env.best_solution.route, best_objective])

    # return [], best_objective


def main(param_file=PARAMETERS_FILE):
    parameters = helper_functions.readJSONFile(param_file)

    folder = DEFAULT_RESULTS_ROOT
    exp_name = 'drl_alns_' + str(parameters["problem_instance"]) + "_" + str(parameters["rseed"])

    best_objective = run_algo(folder, exp_name, **parameters)
    return best_objective


if __name__ == "__main__":
    main()
