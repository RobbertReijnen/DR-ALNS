import pandas as pd
import sys
import math
import random
import numpy as np


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def read_input_cvrp(filename, instance_nr):
    data = pd.read_pickle(filename)
    depot_x = data[instance_nr][0][0]
    depot_y = data[instance_nr][0][1]
    customers_x = [x for x,y in data[instance_nr][1]]
    customers_y = [y for x,y in data[instance_nr][1]]
    demands = data[instance_nr][2]
    capacity = data[instance_nr][3]

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    return len(demands), capacity, distance_matrix, distance_depots, demands

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


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return exact_dist #int(math.floor(exact_dist + 0.5))


def get_nb_trucks(filename):
    begin = filename.rfind("-k")
    if begin != -1:
        begin += 2
        end = filename.find(".", begin)
        return int(filename[begin:end])
    print("Error: nb_trucks could not be read from the file name. Enter it from the command line")
    sys.exit(1)


def compute_route_load(route, demands_data):
    load = 0
    for i in route:
        load += demands_data[i - 1]
    return load


def get_customers_that_can_be_added_to_route(route_load, truck_capacity, unvisited_customers, demands_data):
    unvisited_edgible_customers = []
    for customer in unvisited_customers:
        if route_load + demands_data[customer - 1] <= truck_capacity:
            unvisited_edgible_customers.append(customer)
    return unvisited_edgible_customers


def get_closest_customer_to_add(route, unvisited_edgible_customers, dist_matrix_data, dist_depot_data):
    current_node = route[-1]
    distances = [dist_matrix_data[current_node - 1][unvisited_node - 1] for unvisited_node in
                 unvisited_edgible_customers]
    closest_customer = unvisited_edgible_customers[
        pd.Series(distances).idxmin()]  # NOTE: no -1 because this is an index, not an id
    return closest_customer


def cost_routes(routes, dist_matrix_data, distance_depot_data):
    cost = 0
    for route in routes:
        cost += distance_depot_data[route[0] - 1] + distance_depot_data[route[-1] - 1]
        for i in range(len(route) - 1):
            cost += dist_matrix_data[route[i] - 1][route[i + 1] - 1]
    return cost


def determine_nr_nodes_to_remove(nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=50, omega_plus=0.4):
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
    def __init__(self, num_nodes):
        self.graph = np.full((num_nodes + 1, num_nodes + 1), np.inf, dtype=np.float64)

    def update_edge(self, node_a, node_b, cost):
        # graph is kept single directional
        self.graph[node_a][node_b] = cost

    def get_edge_weight(self, node_a, node_b):
        return self.graph[node_a][node_b]
