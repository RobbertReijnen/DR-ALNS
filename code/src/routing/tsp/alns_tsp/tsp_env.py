def evaluate_solution(route, dist_matrix_data):
    total_distance_travelled = 0

    for i in range(len(route) - 1):
        total_distance_travelled += dist_matrix_data[route[i]][route[i + 1]]
    total_distance_travelled += dist_matrix_data[route[-1]][route[0]]

    return total_distance_travelled


class tspEnv:
    def __init__(self, initial_solution, nb_customers, dist_matrix_data, problem_instance, seed):
        self.nb_customers = nb_customers
        self.dist_matrix_data = dist_matrix_data

        self.seed = seed
        self.problem_instance = problem_instance

        self.route = initial_solution

    def objective(self, best=None):
        score = evaluate_solution(self.route, self.dist_matrix_data)
        return score
