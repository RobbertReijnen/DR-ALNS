import random
import copy
import numpy as np


# --- regret repair
def get_regret_single_insertion(args):
    customer, route, distance_matrix_data = args
    insertions = {}
    distance_matrix_data = np.array(distance_matrix_data)
    for i in range(len(route) + 1):
        updated_route = route[:i] + [customer] + route[i:]
        if i == 0:
            cost_difference = distance_matrix_data[updated_route[-1], updated_route[i]] + distance_matrix_data[
                updated_route[0], updated_route[1]] - distance_matrix_data[updated_route[-1], updated_route[1]]
        elif i == len(route):
            cost_difference = distance_matrix_data[updated_route[i - 1], updated_route[i]] + distance_matrix_data[
                updated_route[i], updated_route[0]] - distance_matrix_data[updated_route[i - 1], updated_route[0]]
        else:
            cost_difference = distance_matrix_data[updated_route[i - 1], updated_route[i]] + distance_matrix_data[
                updated_route[i], updated_route[i + 1]] - distance_matrix_data[
                                  updated_route[i - 1], updated_route[i + 1]]
        insertions[tuple(updated_route)] = cost_difference

    best_insertion = min(insertions, key=insertions.get)
    regret = sorted(list(insertions.values()))[1] - min(insertions.values())
    return best_insertion, regret


def regret_insertion(current, random_state, prob=1.5, **kwargs):
    visited_customers = set(current.route)
    all_customers = set(range(0, current.nb_customers))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    while unvisited_customers:
        insertion_options = {}
        for customer in unvisited_customers:
            best_insertion, regret = get_regret_single_insertion((customer, repaired.route, repaired.dist_matrix_data,))
            insertion_options[best_insertion] = regret

        insertion_option = 0
        while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
            insertion_option += 1
        repaired.route = list(sorted(insertion_options, reverse=True)[insertion_option])

        visited_customers = set([customer for customer in repaired.route])
        unvisited_customers = all_customers - set(visited_customers)
    return repaired


def multi_processing_regret_insertion(current, random_state, prob=1.5, **kwargs):
    visited_customers = set(current.route)
    all_customers = set(range(0, current.nb_customers))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    pool = kwargs.get('pool', None)

    while unvisited_customers:
        args_list = [(customer, repaired.route, current.dist_matrix_data) for customer in unvisited_customers]
        insertions = pool.map(get_regret_single_insertion, args_list)
        insertion_options = {}
        for best_insertion, regret in insertions:
            insertion_options[best_insertion] = regret

        insertion_option = 0
        while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
            insertion_option += 1
        repaired.route = list(sorted(insertion_options, reverse=True)[insertion_option])

        visited_customers = set([customer for customer in repaired.route])
        unvisited_customers = all_customers - set(visited_customers)
    return repaired
