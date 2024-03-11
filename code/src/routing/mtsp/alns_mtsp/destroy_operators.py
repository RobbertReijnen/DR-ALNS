import copy
import random
from routing.mtsp.alns_mtsp.mtsp_helper_functions import determine_nr_nodes_to_remove, NormalizeData


# --- random removal ---
def random_removal(current, random_state, nr_nodes_to_remove=None):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    for i in range(nr_nodes_to_remove):
        if all(len(route) == 2 for route in destroyed_solution.routes):
            break
        eligible_removals = [node for route in destroyed_solution.routes for node in route if len(route)>2]
        node_to_remove = random.choice(eligible_removals)
        for route in destroyed_solution.routes:
            if node_to_remove in route:
                route.remove(node_to_remove)
    return destroyed_solution


# --- relatedness destroy method ---

# see: Shaw - Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems
# see: Santini, Ropke - A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic


def relatedness_removal(current, random_state, nr_nodes_to_remove=None, prob=5):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    eligible_removals = [node for route in destroyed_solution.routes for node in route if len(route) > 2]
    if len(eligible_removals) == 0:
        return destroyed_solution

    node_to_remove = random_state.choice(eligible_removals)
    for route in destroyed_solution.routes:
        if node_to_remove in route:
            route.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        eligible_removals = [node for route in destroyed_solution.routes for node in route if len(route) > 2]
        if len(eligible_removals) == 0:
            return destroyed_solution
        related_nodes = []
        normalized_distances = NormalizeData(destroyed_solution.dist_matrix_data[node_to_remove])
        for route in destroyed_solution.routes:
            for node in route:
                if node in eligible_removals:
                    related_nodes.append((node, normalized_distances[node]))

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(eligible_removals)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        for route in destroyed_solution.routes:
            while node_to_remove in route:
                route.remove(node_to_remove)
    return destroyed_solution


# --- neighbor/history graph removal
# see: A unified heuristic for a large class of Vehicle Routing Problems with Backhauls
def neighbor_graph_removal(current, random_state, nr_nodes_to_remove=None, prob=5):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    eligible_removals = [node for route in destroyed_solution.routes for node in route if len(route) > 2]
    if len(eligible_removals) == 0:
        return destroyed_solution

    values = {}
    for route in destroyed_solution.routes:
        for i in range(len(route)):
            if route[i] in eligible_removals:
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(0, route[i]) + current.graph.get_edge_weight(
                        route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], 0)
                else:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], route[i + 1])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove and len(eligible_removals) > 0:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)

        # select the node to remove
        removal_option = 0
        while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        # remove the node from its route
        for route in destroyed_solution.routes:
            if node_to_remove in route:
                route.remove(node_to_remove)
                removed_nodes.append(node_to_remove)

                values.pop(node_to_remove)
                for i in range(len(route)):
                    if i == 0:
                        values[route[i]] = current.graph.get_edge_weight(0, route[
                            i]) + current.graph.get_edge_weight(route[i], route[1])
                    elif i == len(route) - 1:
                        values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                            i]) + current.graph.get_edge_weight(route[i], 0)
                    else:
                        values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                            i]) + current.graph.get_edge_weight(route[i], route[i + 1])

                eligible_removals = [node for route in destroyed_solution.routes for node in route if len(route) > 2]

                keys_to_remove = []
                for node in values.keys():
                    if node not in eligible_removals:
                        keys_to_remove.append(node)

                for node in keys_to_remove:
                    values.pop(node)
    return destroyed_solution
