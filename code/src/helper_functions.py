import pathlib
import csv
import json
from pathlib import Path


def write_output(folder, exp_name, problem_instance, seed, iterations, solution, best_objective, instance_file):
    """Save outputs in files"""
    output_dir = folder
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Final pop
    with open(output_dir + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['problem_instance', 'rseed', 'iterations', 'solution', 'best_objective', 'instance_file'])
        writer.writerow([problem_instance, seed, iterations, solution, best_objective, instance_file])


def writeJSONfile(data, path):
    with open(path, "w") as write_file:
        json.dump(data, write_file, indent=4)
        write_file.write("\n")


def readJSONFile(file, check_if_exists=False):
    """This function reads any json file and returns a dictionary."""
    if (not Path(file).is_file()) and check_if_exists:
        return None
    with open(file) as f:
        data = json.load(f)
    return data