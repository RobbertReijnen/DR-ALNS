import math
import numpy as np


def dist_l2_closest_integer(x1, x2):
    """Compute the L2-norm (Euclidean) distance between two points.
    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.
    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    x_diff = x2[0] - x1[0]
    y_diff = x2[1] - x1[1]
    return int(math.sqrt(x_diff * x_diff + y_diff * y_diff) + .5)


def dist_l2(x1, x2, rd=5):
    """Compute the L2-norm (Euclidean) distance between two points.
    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    x_diff = x2[0] - x1[0]
    y_diff = x2[1] - x1[1]
    return round(math.sqrt(x_diff * x_diff + y_diff * y_diff), rd)


def make_dist_matrix(points, dist=dist_l2_closest_integer, to_integer=True, rd=4):
    """Compute a distance matrix for a set of points.
    Uses function 'dist' to calculate distance between
    any two points.
    """
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            x1 = points[i]
            x2 = points[j]
            x1 = x1 * 10 ** rd
            x2 = x2 * 10 ** rd
            dist_matrix[i, j] = dist(x1, x2)
            dist_matrix[j, i] = dist_matrix[i, j]
    if to_integer:
        dist_matrix = dist_matrix.astype(int)
    else:
        dist_matrix = dist_matrix / 10 ** rd

    return dist_matrix


def tsp_tour_cost(tour, cost_matrix):
    cost = 0

    for i in range(len(tour) - 1):
        node = int(tour[i])
        succ = int(tour[i + 1])
        cost += cost_matrix[node][succ]

    return cost


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