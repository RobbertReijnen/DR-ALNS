import copy
import random
from routing.tsp.alns_tsp.tsp_helper_functions import determine_nr_nodes_to_remove, NormalizeData


# --- random removal ---

def random_removal(current, random_state, nr_nodes_to_remove=None, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    customers = [i for i in range(0, current.nb_customers)]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    nodes_to_remove = random.sample(customers, nr_nodes_to_remove)
    destroyed_solution.route = [customer for customer in destroyed_solution.route if customer not in nodes_to_remove]

    return destroyed_solution


# --- relatedness destroy method ---

# see: Shaw - Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems
# see: Santini, Ropke - A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic

def relatedness_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [node for node in destroyed_solution.route]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    node_to_remove = random.choice(visited_customers)
    destroyed_solution.route.remove(node_to_remove)
    visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        related_nodes = []
        normalized_distances = NormalizeData(destroyed_solution.dist_matrix_data[node_to_remove]) # TODO: REDUNDANT?
        for node in destroyed_solution.route:
            related_nodes.append((node, normalized_distances[node]))

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        destroyed_solution.route.remove(node_to_remove)
        visited_customers.remove(node_to_remove)
    return destroyed_solution


# --- neighbor/history graph removal
# see: A unified heuristic for a large class of Vehicle Routing Problems with Backhauls
def neighbor_graph_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    #TODO: should only work if there is a x nr of customers still in solution
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    values = {}
    for i in range(1, len(destroyed_solution.route)):
        if i != len(destroyed_solution.route)-1:
            values[destroyed_solution.route[i]] = current.graph.get_edge_weight(destroyed_solution.route[i - 1],
                                                              destroyed_solution.route[i]) + current.graph.get_edge_weight(
                destroyed_solution.route[i], destroyed_solution.route[i + 1])
        else:
            values[destroyed_solution.route[i]] = current.graph.get_edge_weight(destroyed_solution.route[i-1],
                                                              destroyed_solution.route[i]) + current.graph.get_edge_weight(
                destroyed_solution.route[i], destroyed_solution.route[0])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        # select the node to remove
        removal_option = 0
        while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        destroyed_solution.route.remove(node_to_remove)
        removed_nodes.append(node_to_remove)

        values.pop(node_to_remove)

        for i in range(1, len(destroyed_solution.route)):
            if i != len(destroyed_solution.route) - 1:
                values[destroyed_solution.route[i]] = current.graph.get_edge_weight(destroyed_solution.route[i - 1],
                                                                                    destroyed_solution.route[
                                                                                        i]) + current.graph.get_edge_weight(
                    destroyed_solution.route[i], destroyed_solution.route[i + 1])
            else:
                values[destroyed_solution.route[i]] = current.graph.get_edge_weight(destroyed_solution.route[i - 1],
                                                                                    destroyed_solution.route[
                                                                                        i]) + current.graph.get_edge_weight(
                    destroyed_solution.route[i], destroyed_solution.route[0])
    return destroyed_solution
