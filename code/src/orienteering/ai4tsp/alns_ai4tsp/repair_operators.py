import copy
import numpy as np
import random

from orienteering.ai4tsp.alns_ai4tsp.ai4tsp_helper_functions import tour_check

NR_INTERMEDIATE_SOLUTION_EVALUATIONS = 2


def get_best_distance_insertion_for_node(node, tour, adj):
    """returns insertion of index and node which results in least addition distance"""
    tour = np.array(tour)
    adj = np.array(adj)

    # Compute distances using NumPy broadcasting
    predecessor_nodes = tour[:-1]
    successor_nodes = tour[1:]
    distances = adj[node - 1, predecessor_nodes - 1] + adj[node - 1, successor_nodes - 1] - adj[
        predecessor_nodes - 1, successor_nodes - 1]

    # Find index of minimum distance using argmin
    min_index = np.argmin(distances)

    # Return index of position to insert node
    return min_index + 1


def random_best_distance_repair(current, random_state, **kwargs):
    """Randomly select a nr of nodes and add, according to a random generated sequence, at its least expensive position
    (in terms of distance) """
    curr_obj = current.objective()
    visited = current.route[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    for node in nodes_to_include:
        # Find best position to insert node.
        index = get_best_distance_insertion_for_node(node, current.route, current.adj)
        candidate = copy.deepcopy(current)
        candidate.route.insert(index, node)
        cand_obj = candidate.objective()
        if cand_obj < curr_obj:
            curr_obj = cand_obj
            current = copy.deepcopy(candidate)

    # print('return distance', current.objective())
    return current


# -------- price repair ----------------------
def multiprocess_best_prize_insertions_for_node(args):
    nodes, tour, inx, node, x, adj = args

    new_tour = tour[:inx] + [node] + tour[inx:]

    total_reward, total_pen = 0, 0
    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        tour_time, rewards, pen, feas = tour_check(new_tour, x, adj, -1.0, -1.0, len(nodes))
        total_reward += rewards
        total_pen += pen
    score = - (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    return {tuple(new_tour): score}


def get_best_prize_insertion_for_node(node, nodes, tour, input_score, adj, x, pool=None):
    """returns insertion of index and node which results in least addition distance"""
    best_new_tour, best_score = None, input_score

    args_list = [(nodes, tour, inx, node, x, adj) for inx in range(1, len(tour))]
    for arg in args_list:
        item = multiprocess_best_prize_insertions_for_node(arg)
        for new_tour, score in item.items():
            if best_score > score:
                best_new_tour = list(new_tour)
                best_score = score

    # DOES NOT BRING THE HOPED BENEFITS
    # args_list = [(nodes, tour, inx, node, x, adj) for inx in range(1, len(tour))]
    # multiprocess_results = pool.map(multiprocess_best_prize_insertions_for_node, args_list)
    #
    # for item in multiprocess_results:
    #     for new_tour, score in item.items():
    #         if best_score > score:
    #             best_new_tour = list(new_tour)
    #             best_score = score

    if best_new_tour is None:
        return tour
    else:
        return best_new_tour


def random_best_prize_repair(current, random_state, **kwargs):
    """Randomly select a nr of nodes and add, according to a random generated sequence, the nodes sequentially to their
    best positions (in terms of accumulated rewards) """
    current = copy.copy(current)
    curr_obj = current.objective()

    visited = current.route[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    pool = kwargs.get('pool', None)
    for node in nodes_to_include:
        candidate = copy.copy(current)
        cand_obj = curr_obj
        candidate.route = get_best_prize_insertion_for_node(node, candidate.nodes, candidate.route, cand_obj,
                                                            candidate.adj, candidate.x, pool)
        cand_obj = candidate.objective()
        if cand_obj < curr_obj:
            curr_obj = cand_obj
            current = copy.copy(candidate)

    # print('return price', current.objective())
    return current


# -------- ratio repair ----------------------
def multiprocess_best_ratio_insertions_for_node(args):
    nodes, tour, current_score, current_tour_time, inx, node, x, adj = args
    new_tour = tour[:inx] + [node] + tour[inx:]

    total_tour_time, total_reward, total_pen = 0, 0, 0
    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        tour_time, rewards, pen, feas = tour_check(new_tour, x, adj, -1.0, -1.0, len(nodes))
        total_tour_time += tour_time
        total_reward += rewards
        total_pen += pen
    tour_time = total_tour_time / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    score = - (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    if tour_time - current_tour_time == 0:
        ratio = 0
    elif score < 0 and current_score < 0.00000001 and score < current_score:
        if tour_time < current_tour_time:  # free travelling time:
            ratio = abs(score - current_score)
        else:
            ratio = abs(score - current_score) / (tour_time - current_tour_time)
    else:
        ratio = 0

    return {tuple(new_tour): {'score': score, 'ratio': ratio, 'time': tour_time}}


def get_best_ratio_insertion_for_node(node, nodes, tour, input_score, input_time, adj, x, pool=None):
    best_new_tour, best_ratio, best_score = None, 0, 0
    for inx in range(1, len(tour)):
        args = (nodes, tour, input_score, input_time, inx, node, x, adj)
        item = multiprocess_best_ratio_insertions_for_node(args)
        for new_tour, result in item.items():
            if result['ratio'] > best_ratio:
                best_new_tour = list(new_tour)
                best_ratio = result['ratio']
                best_time = result['time']

    # DOES NOT BRING THE HOPED BENEFITS
    # best_new_tour, best_ratio = None, 0
    # args_list = [(nodes, tour, input_score, input_time, inx, node, x, adj) for inx in range(1, len(tour))]
    # multiprocess_results = pool.map(multiprocess_best_ratio_insertions_for_node, args_list)
    #
    # for item in multiprocess_results:
    #     for new_tour, result in item.items():
    #         if best_ratio > result['ratio']:
    #             best_new_tour = list(new_tour)
    #             best_ratio = result['ratio']
    #             best_score = result['score']
    #             best_time = result['time']

    # print(best_new_tour, best_score)
    if best_new_tour is None:
        return None, None
    else:
        return best_new_tour, best_time,


def random_best_ratio_repair(current, random_state, **kwargs):
    """Find the best insertions to be done (in terms of additional reward/additional distance ratio)"""

    pool = kwargs.get('pool', None)
    current = copy.deepcopy(current)

    visited = current.route[:-1]  # all visited nodes
    not_visited = [x for x in current.nodes if x not in visited]  # all unvisited nodes

    # only add a random number of the nodes that are not in the destroyed solution back
    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())  # shuffled nodes to visit

    # get score and route time of the current (destroyed) route
    total_route_time, total_reward, total_pen = 0, 0, 0

    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        route_time, rewards, pen, feas = tour_check(current.route, current.x, current.adj, -1.0, -1.0,
                                                    len(current.nodes))
        total_reward += rewards
        total_pen += pen
        total_route_time += route_time

    curr_obj = - (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    curr_route_time = total_route_time / NR_INTERMEDIATE_SOLUTION_EVALUATIONS

    for node in nodes_to_include:
        candidate = copy.copy(current)
        candidate.route, cand_route_time = get_best_ratio_insertion_for_node(node, current.nodes, current.route,
                                                                             curr_obj, curr_route_time, current.adj,
                                                                             current.x, pool)
        if candidate.route != None:
            cand_obj = candidate.objective()
            if cand_obj < curr_obj:
                curr_obj = cand_obj
                current = copy.copy(candidate)
                curr_route_time = cand_route_time

    # print(current_obj)
    # print('return ratio', current.objective())
    return current


# -------- regret repair ----------------------

# def get_regret_single_insertion(route, customer, nr_customers, adj, x):
def get_regret_single_insertion(args):
    route, customer, nr_customers, adj, x = args
    insertions = {}
    for i in range(1, len(route)):
        updated_route = route[:i] + [customer] + route[i:]
        # evaluate new route
        total_reward, total_pen = 0, 0
        for j in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
            tour_time, rewards, pen, feas = tour_check(updated_route, x, adj, -1.0, -1.0, nr_customers)
            total_reward += rewards
            total_pen += pen
        score = (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
        if score > 0:
            insertions[tuple(updated_route)] = score

    if len(insertions) == 1:
        best_insertion = min(insertions, key=insertions.get)
        return best_insertion, 0

    elif len(insertions) > 1:
        best_insertion = max(insertions, key=insertions.get)  # list(map(list, min(insertions, key=insertions.get)))

        if len(set(insertions.values())) == 1:  # when all options are of equal value:
            regret = 0
        else:
            regret = max(insertions.values()) - sorted(insertions.values(), reverse=True)[1]
        return best_insertion, regret
    else:
        # no benefitial insertions possible for this customer
        return None, None


def regret_insertion(current, random_state, prob=1.5, **kwargs):
    repaired_solution = copy.deepcopy(current)
    visited_customers = list(repaired_solution.route[:-1])
    all_customers = repaired_solution.nodes
    unvisited_customers = [x for x in all_customers if x not in visited_customers]
    pool = kwargs.get('pool', None)

    while True:
        insertion_options = {}
        route = repaired_solution.route[:]
        for customer in unvisited_customers:
            args = (route, customer, len(all_customers), repaired_solution.adj, repaired_solution.x)
            best_insertion, regret = get_regret_single_insertion(args)
            if best_insertion is not None:
                insertion_options[best_insertion] = regret

        # multicore processing is always faster for this repair!
        # insertion_options = {}
        # args = [(repaired_solution.route, customer, len(all_customers), repaired_solution.adj, repaired_solution.x) for customer in unvisited_customers]
        # results = pool.map(get_regret_single_insertion, args)
        # for customer, (best_insertion, regret) in zip(unvisited_customers, results):
        #     if best_insertion is not None:
        #         insertion_options[best_insertion] = regret

        if len(insertion_options) > 0:
            insertion_option = 0
            while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
                insertion_option += 1

            repaired_solution.route = list(sorted(insertion_options, reverse=False)[insertion_option])
            visited_customers = list(repaired_solution.route[:-1])
            unvisited_customers = [x for x in all_customers if x not in visited_customers]

        else:
            return repaired_solution


def beam_search(current, random_state, **kwargs):
    beam_width = 10
    repaired_solution = copy.deepcopy(current)
    print('destroyed', repaired_solution.route)
    all_customers = repaired_solution.nodes
    all_paths = [[current.route, current.objective()]]

    while True:
        temp_paths = []
        for route, objective in all_paths:
            for i in range(len(route)-1):
                for node in all_customers:
                    if node not in route:
                        new_solution = route[:i + 1] + [node] + route[i + 1:]
                        total_reward, total_pen = 0, 0
                        for j in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
                            tour_time, rewards, pen, feas = tour_check(new_solution, current.x, current.adj, -1.0, -1.0, len(current.nodes))
                            total_reward += rewards
                            total_pen += pen
                        new_objective = - (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
                        if new_objective <= objective:
                            temp_paths.append([new_solution, new_objective])

        # If no new paths were added, break the loop
        if len(temp_paths) == 0:
            break

        # Sorting paths based on their cost and taking only top 'beam_width' paths
        all_paths = sorted(temp_paths, key=lambda x: x[1])[:beam_width]


    repaired_solution.route = all_paths[0][0]
    print('repaired', repaired_solution.route)
    return repaired_solution