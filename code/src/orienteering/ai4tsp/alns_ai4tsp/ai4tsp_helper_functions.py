import json
import copy
import csv
import pandas as pd
import numpy as np

from pathlib import Path

# --- FILE READING AND WRITING ------------------------------
def write_output(folder, exp_name, problem_instance, seed, iterations, solution, best_objective):
    """Save outputs in files"""
    output_dir = folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Final pop
    with open(output_dir + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([problem_instance, seed, iterations, solution, best_objective])


def readJSONFile(file, check_if_exists=False):
    """This function reads any json file and returns a dictionary."""
    if (not Path(file).is_file()) and check_if_exists:
        return None
    with open(file) as f:
        data = json.load(f)
    return data

def read_instance(x_path, adj_path):
    x_df = pd.read_csv(x_path, sep=',')
    adj_df = pd.read_csv(adj_path, sep=',')

    x, adj = x_df.to_numpy(), adj_df.to_numpy()

    return x, adj, Path(x_path).stem


def update_neighbor_graph(current, route, new_route_quality):
    graph = copy.copy(current.graph)
    edge_weights = [graph.get_edge_weight(route[i-1], route[i]) for i in range(1, len(route))]
    updated_edges = [(route[i-1], route[i]) for i in range(1, len(route)) if new_route_quality < edge_weights[i-1]]
    for edge in updated_edges:
        graph.update_edge(edge[0], edge[1], new_route_quality)
    return graph

import math
import numpy as np

def tour_check(tour, x, time_matrix, maxT_pen, tw_pen, n_nodes):
    """
    Calculate a tour times and the penalties for constraint violation
    """
    tw_high = x[:, -3]
    tw_low = x[:, -4]
    prizes = x[:, -2]
    maxT = x[0, -1]

    feas = True
    return_to_depot = False
    tour_time = 0
    rewards = 0
    pen = 0

    for i in range(len(tour) - 1):

        node = int(tour[i])
        if i == 0:
            assert node == 1, 'A tour must start from the depot - node: 1'

        succ = int(tour[i + 1])
        time = time_matrix[node - 1][succ - 1]
        noise = np.random.randint(1, 101, size=1)[0]/100
        tour_time += np.round(noise * time, 2)
        if tour_time > tw_high[succ - 1]:
            feas = False
            # penalty added for each missed tw
            pen += tw_pen
        elif tour_time < tw_low[succ - 1]:
            tour_time += tw_low[succ - 1] - tour_time
            rewards += prizes[succ - 1]
        else:
            rewards += prizes[succ - 1]

        if succ == 1:
            return_to_depot = True
            break

    if not return_to_depot:
        raise Exception('A tour must reconnect back to the depot - node: 1')

    if tour_time > maxT:
        # penalty added for each
        pen += maxT_pen * n_nodes
        feas = False

    return tour_time, rewards, pen, feas

class NeighborGraph:
    def __init__(self, num_nodes: int):
        self.graph = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)

    def update_edge(self, node_a: int, node_b: int, cost: float):
        self.graph[node_a-1][node_b-1] = cost

    def get_edge_weight(self, node_a: int, node_b: int) -> float:
        return self.graph[node_a-1][node_b-1]