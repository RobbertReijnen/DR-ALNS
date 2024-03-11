import pandas as pd
import sys
import math
import random
import numpy as np


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def read_input_mtsp(filename, instance_nr):
    data = pd.read_pickle(filename)
    coordinates_x = [x for x,y in data[instance_nr]]
    coordinates_y = [y for x,y in data[instance_nr]]

    distance_matrix = compute_distance_matrix(coordinates_x, coordinates_y)
    nr_vehicles = 4

    nb_customers = len(coordinates_x) - 1  # -1 for depot
    return nb_customers, distance_matrix, nr_vehicles, coordinates_x, coordinates_y

# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = np.zeros((nb_customers, nb_customers))
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return exact_dist #int(math.floor(exact_dist + 0.5))


def determine_nr_nodes_to_remove(nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=30, omega_plus=0.4):
    n_plus = min(omega_bar_plus, omega_plus * nb_customers)
    n_minus = min(n_plus, max(omega_bar_minus, omega_minus * nb_customers))
    r = random.randint(round(n_minus), round(n_plus))
    return r


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def update_neighbor_graph(current, new_routes, new_routes_quality):
    for route in new_routes:
        prev_node = 0
        for i in range(len(route)):
            curr_node = route[i]
            prev_edge_weight = current.graph.get_edge_weight(prev_node, curr_node)
            if new_routes_quality < prev_edge_weight:
                current.graph.update_edge(prev_node, curr_node, new_routes_quality)
            prev_node = curr_node
        prev_edge_weight = current.graph.get_edge_weight(prev_node, 0)
        if new_routes_quality < prev_edge_weight:
            current.graph.update_edge(prev_node, 0, new_routes_quality)
    return current.graph


class NeighborGraph:
    def __init__(self, nb_customers):
        self.graph = np.full((nb_customers + 1, nb_customers + 1), np.inf, dtype=np.float64)

    def update_edge(self, node_a, node_b, cost):
        # graph is kept single directional
        self.graph[node_a][node_b] = cost

    def get_edge_weight(self, node_a, node_b):
        return self.graph[node_a][node_b]
