def evaluate_solution(routes, dist_matrix_data, nr_vehicles):
    total_distance_travelled = 0

    for route in routes:
        total_distance_travelled += dist_matrix_data[0, route[0]] + dist_matrix_data[route[-1], 0]
        for i in range(len(route) - 1):
            total_distance_travelled += dist_matrix_data[route[i]][route[i + 1]]
    if len(routes) != nr_vehicles:
        print('routes does not match nr of vehicles')

    for route in routes:
        if len(route) < 2:
            print('size of routes is incorrect')

    return total_distance_travelled


class mtspEnv:

    def __init__(self, initial_solution, nb_customers, dist_matrix_data, nr_vehicles, problem_instance, seed):

        self.routes = initial_solution
        self.nb_customers = nb_customers
        self.dist_matrix_data = dist_matrix_data
        self.nr_vehicles = nr_vehicles

        self.problem_instance = problem_instance
        self.seed = seed

    def objective(self, **kwargs):
        score = evaluate_solution(self.routes, self.dist_matrix_data, self.nr_vehicles)
        return score
