import random
import copy


# --- regret repair
def get_regret_single_insertion(routes, customer, distance_matrix_data):
    insertions = {}
    for route_idx in range(len(routes)):
        for i in range(len(routes[route_idx]) + 1):
            updated_route = routes[route_idx][:i] + [customer] + routes[route_idx][i:]
            updated_routes = routes[:route_idx] + [updated_route] + routes[route_idx + 1:]
            if i == 0:
                cost_difference = distance_matrix_data[0, updated_route[0]] + distance_matrix_data[updated_route[0], updated_route[1]] - distance_matrix_data[0, updated_route[1]]
            elif i == len(routes[route_idx]):
                cost_difference = distance_matrix_data[updated_route[-1], 0] + distance_matrix_data[updated_route[i-1], updated_route[i]] - distance_matrix_data[updated_route[i-1], 0]
            else:
                cost_difference = distance_matrix_data[updated_route[i-1], updated_route[i]] + distance_matrix_data[updated_route[i], updated_route[i+1]] - distance_matrix_data[updated_route[i-1], updated_route[i+1]]

            insertions[tuple(map(tuple, updated_routes))] = cost_difference

    if len(insertions) == 1:
        best_insertion = min(insertions, key=insertions.get)
        return best_insertion, 0

    else:
        best_insertion = min(insertions, key=insertions.get)

        if len(set(insertions.values())) == 1:  # when all options are of equal value:
            regret = 0
        else:
            regret = sorted(list(insertions.values()))[1] - min(insertions.values())
        return best_insertion, regret


def regret_insertion(current, random_state, prob=1.5, **kwargs):
    visited_customers = [customer for route in current.routes for customer in route]
    all_customers = set(range(1, current.nb_customers + 1))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    while unvisited_customers:
        insertion_options = {}
        for customer in unvisited_customers:
            best_insertion, regret = get_regret_single_insertion(repaired.routes, customer, repaired.dist_matrix_data)
            insertion_options[best_insertion] = regret

        if not insertion_options:
            repaired.routes.append([random.choice(list(unvisited_customers))])
        else:
            insertion_option = 0
            while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
                insertion_option += 1
            repaired.routes = list(map(list, sorted(insertion_options, reverse=True)[insertion_option]))

        visited_customers = [customer for route in repaired.routes for customer in route]
        unvisited_customers = all_customers - set(visited_customers)
    return repaired
