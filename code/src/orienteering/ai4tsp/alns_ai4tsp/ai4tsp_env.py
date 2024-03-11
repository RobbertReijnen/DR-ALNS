import numpy as np
from orienteering.ai4tsp.alns_ai4tsp.ai4tsp_helper_functions import tour_check, NeighborGraph


class ai4tspEnv:

    def __init__(self, nodes: list[int], init_route: list[int], x: list[float], adj: list[list[float]],
                 problem_instance: str, seed: int):
        self.nodes = nodes
        self.route = init_route

        self.seed = seed
        self.problem_instance = problem_instance
        self.x = x
        self.adj = adj
        self.graph = NeighborGraph(len(self.nodes))
        self.maxT_pen = -1.0 #DEFAULT FROM COMPETITION
        self.tw_pen = -1.0 #DEFAULT FROM COMPETITION

    def update_edge_weight(self, node_a: int, node_b: int, cost: float):
        self.graph.update_edge(node_a - 1, node_b - 1, cost)

    def check_solution(self, sol: list[int]) -> tuple[float, float, float, bool]:
        assert len(sol) == len(self.x) + 1, f"len(sol) = {len(sol)}, n_nodes+1 = {len(self.x) + 1}"
        assert len(sol) == len(set(sol)) + 1
        tour_time, rewards, pen, feas = tour_check(sol, self.x, self.adj, self.maxT_pen, self.tw_pen, len(self.nodes))
        return tour_time, rewards, pen, feas

    def evaluate_individual_solution(self, solution: list[int], eval=30) -> float:
        total_reward, total_pen = 0, 0
        for _ in range(eval):
            route_time, rewards, pen, feas = self.check_solution(solution)
            total_reward += rewards
            total_pen += pen
        score = - (total_reward + total_pen) / eval
        if score == -0.0:
            return 0
        return score

    def objective(self, best=False) -> float:
        # all nodes must be present in the evaluation function: [1,2,3,1,5] --> will stop evaluation after second '1'
        route = self.route
        unvisited_nodes = [x for x in self.nodes if x not in route]
        evaluation_tour = route + unvisited_nodes

        if best:
            score = self.evaluate_individual_solution(evaluation_tour, eval=100)
        else:
            score = self.evaluate_individual_solution(evaluation_tour, eval=30)
        return score