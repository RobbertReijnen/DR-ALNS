def evaluate_solution(routes, truck_capacity, dist_matrix_data, dist_depot_data, demands_data):
    total_distance_travelled = 0

    for route in routes:
        total_distance_travelled += dist_depot_data[route[0] - 1] + dist_depot_data[route[-1] - 1]
        for i in range(len(route) - 1):
            total_distance_travelled += dist_matrix_data[route[i] - 1][route[i + 1] - 1]

    # can be removed in deployment, just for testing
    # for route in routes:
    #     if compute_route_load(route, demands_data) > truck_capacity:
    #         print('TOO MUCH LOAD FOR TRUCK')
    return total_distance_travelled


class cvrpEnv:

    def __init__(self, initial_solution, nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data, problem_instance, seed):
        self.nb_customers = nb_customers
        self.truck_capacity = truck_capacity
        self.dist_matrix_data = dist_matrix_data
        self.dist_depot_data = dist_depot_data
        self.demands_data = demands_data

        self.seed = seed
        self.problem_instance = problem_instance

        self.routes = initial_solution

    def objective(self, best=False):
        score = evaluate_solution(self.routes, self.truck_capacity, self.dist_matrix_data, self.dist_depot_data, self.demands_data)
        return score
