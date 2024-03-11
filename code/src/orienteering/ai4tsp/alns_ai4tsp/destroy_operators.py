import copy
from routing.cvrp.alns_cvrp.cvrp_helper_functions import NormalizeData
import random


def random_removal(current, random_state, degree_of_destruction=None, **kwargs):
    if current.route == [1, 1]:
        return current

    nodes = current.route[:-1]
    destroyed_solution = copy.copy(current)

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(nodes) - 1))

    idx_to_remove = random_state.choice(range(1, len(nodes)), nr_nodes_to_remove, replace=False)
    destroyed_solution.route = [i for j, i in enumerate(current.route) if j not in idx_to_remove]
    # print('  ')

    return destroyed_solution


def relatedness_removal(current, random_state, prob=5, degree_of_destruction=None, **kwargs):
    if current.route == [1, 1]:
        return current

    destroyed_solution = copy.deepcopy(current)
    visited_customers = list(destroyed_solution.route[1:-1])

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(visited_customers)))

    node_to_remove = random_state.choice(visited_customers)
    while node_to_remove in destroyed_solution.route:
        destroyed_solution.route.remove(node_to_remove)
        visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        normalized_distances = NormalizeData(current.adj[node_to_remove - 1])
        related_nodes = [(node, normalized_distances[node - 1]) for node in visited_customers]

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        while node_to_remove in destroyed_solution.route:
            destroyed_solution.route.remove(node_to_remove)
            visited_customers.remove(node_to_remove)
    return destroyed_solution


# def neighbor_graph_removal(current, random_state, degree_of_destruction=None, prob=5):
#     if current.route == [1, 1]:
#         return current
#     destroyed_solution = copy.copy(current)
#     visited_customers = list(destroyed_solution.route[1:-1])
#
#     nr_nodes_to_remove = max(1, round(degree_of_destruction * len(visited_customers)))
#
#     values = {}
#     route = destroyed_solution.route[1:-1]
#
#     if len(route) == 1:
#         values[route[0]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(route[0],
#                                                                                                           1)
#     else:
#         for i in range(len(route)):
#             if i == 0:
#                 values[route[i]] = current.graph.get_edge_weight(1, route[i]) + current.graph.get_edge_weight(
#                     route[i], route[1])
#             elif i == len(route) - 1:
#                 values[route[i]] = current.graph.get_edge_weight(route[i - 1],
#                                                                  route[i]) + current.graph.get_edge_weight(
#                     route[i], 1)
#             else:
#                 values[route[i]] = current.graph.get_edge_weight(route[i - 1],
#                                                                  route[i]) + current.graph.get_edge_weight(
#                     route[i], route[i + 1])
#
#     removed_nodes = []
#     while len(removed_nodes) < nr_nodes_to_remove:
#         # sort the nodes based on their neighbor graph scores in descending order
#         sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=False)
#         # select the node to remove
#         removal_option = 0
#         while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
#             removal_option += 1
#         node_to_remove, score = sorted_nodes[removal_option]
#
#         # remove the node from the route
#         index_removed_node = route.index(node_to_remove)
#         route.remove(node_to_remove)
#         destroyed_solution.route.remove(node_to_remove)
#         removed_nodes.append(node_to_remove)
#         values.pop(node_to_remove)
#
#         # update values of surrounding customers, this would have been much easier with the 1 in the route....:
#         if len(route) == 1:
#             values[route[0]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(
#                 route[0], 1)
#         elif len(route) > 1:
#             if index_removed_node == 0:
#                 values[route[index_removed_node]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(route[0], route[1])
#             elif index_removed_node == 1:
#                 values[route[0]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(route[0], route[1])
#                 if len(route) == 2:
#                     values[route[1]] = current.graph.get_edge_weight(route[0], route[1]) + current.graph.get_edge_weight(route[1], 1)
#                 else:
#                     values[route[1]] = current.graph.get_edge_weight(route[0],route[1]) + current.graph.get_edge_weight(route[1], route[2])
#             elif index_removed_node == len(route):
#                 values[route[index_removed_node-1]] = current.graph.get_edge_weight(route[index_removed_node - 2], route[index_removed_node-1]) + current.graph.get_edge_weight(route[index_removed_node-1], 1)
#             else:
#                 values[route[index_removed_node-1]] = current.graph.get_edge_weight(route[index_removed_node - 2], route[index_removed_node-1]) + current.graph.get_edge_weight(route[index_removed_node-1], route[index_removed_node])
#                 if len(route) != index_removed_node+1:
#                     values[route[index_removed_node - 1]] = current.graph.get_edge_weight(route[index_removed_node - 1], route[index_removed_node]) + current.graph.get_edge_weight(route[index_removed_node], route[index_removed_node+1])
#                 else:
#                     values[route[index_removed_node - 1]] = current.graph.get_edge_weight(route[index_removed_node - 1], route[index_removed_node]) + current.graph.get_edge_weight(route[index_removed_node], 1)
#     return destroyed_solution

# OLD ONE, NEW SHOULD BE FASTER --> TODO...
def neighbor_graph_removal(current, random_state, degree_of_destruction=None, prob=5, **kwargs):
    if current.route == [1, 1]:
        return current
    destroyed_solution = copy.deepcopy(current)
    visited_customers = list(destroyed_solution.route[1:-1])

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(visited_customers)))

    values = {}
    route = destroyed_solution.route[1:-1]

    if len(route) == 1:
        values[route[0]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(route[0],
                                                                                                      1)
    else:
        for i in range(len(route)):
            if i == 0:
                values[route[i]] = current.graph.get_edge_weight(1, route[i]) + current.graph.get_edge_weight(
                    route[i], route[1])
            elif i == len(route) - 1:
                values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                 route[i]) + current.graph.get_edge_weight(
                    route[i], 1)
            else:
                values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                 route[i]) + current.graph.get_edge_weight(
                    route[i], route[i + 1])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=False)
        # select the node to remove
        removal_option = 0
        while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        # remove the node from the route
        route.remove(node_to_remove)
        destroyed_solution.route.remove(node_to_remove)
        removed_nodes.append(node_to_remove)
        values.pop(node_to_remove)

        if len(route) == 0:
            continue

        elif len(route) == 1:
            values[route[0]] = current.graph.get_edge_weight(1, route[0]) + current.graph.get_edge_weight(
                route[0] - 1, 1)
        else:
            for i in range(len(route)):
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(1, route[
                        i]) + current.graph.get_edge_weight(route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                        i]) + current.graph.get_edge_weight(route[i], 1)
                else:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                        i]) + current.graph.get_edge_weight(route[i], route[i + 1])

    return destroyed_solution
