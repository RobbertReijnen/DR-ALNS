import csv
from pathlib import Path
from rl.environments.tsp_AlnsEnv_LSA1 import tspAlnsEnv_LSA1
import helper_functions

from stable_baselines3 import PPO

DEFAULT_RESULTS_ROOT = "single_runs/"
PARAMETERS_FILE = "configs/drl_alns_tsp_debug.json"


def run_algo(folder, exp_name, client=None, **kwargs):
    instance_nr = kwargs['instance_nr']
    seed = kwargs['rseed']
    iterations = kwargs['iterations']

    base_path = Path(__file__).parent.parent.parent
    instance_file = str(base_path.joinpath(kwargs['instance_file']))
    model_path = base_path / kwargs['model_directory'] / 'model'
    model = PPO.load(model_path)

    parameters = {'environment': {'iterations': iterations, 'instance_nr': [instance_nr], 'instance_file': instance_file}}
    env = tspAlnsEnv_LSA1(parameters)
    env.run(model)
    best_objective = env.best_solution.objective()
    print("best_obj", best_objective)

    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(folder + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'instance_file'])
        writer.writerow([instance_nr, seed, iterations, env.best_solution.route, best_objective, kwargs['instance_file']])

    return [], best_objective


def main(param_file=PARAMETERS_FILE):
    parameters = helper_functions.readJSONFile(param_file)

    folder = DEFAULT_RESULTS_ROOT
    exp_name = 'drl_alns_' + str(parameters["instance_nr"]) + "_" + str(parameters["rseed"])

    best_objective = run_algo(folder, exp_name, **parameters)
    return best_objective


if __name__ == "__main__":
    main()