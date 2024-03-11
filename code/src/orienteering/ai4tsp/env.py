import numpy as np
import random
import orienteering.ai4tsp.op_utils.instance as u_i
import orienteering.ai4tsp.op_utils.op as u_o


class Env:
    maxT_pen = -1.0
    tw_pen = -1.0

    def __init__(self, n_nodes=50, seed=None, from_file=False, x_path=None, adj_path=None):

        self.x = None
        self.adj = None
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.sim_counter = 0
        self.name = None
        if from_file:
            self.x, self.adj, self.instance_name = u_i.read_instance(x_path, adj_path)
            self.n_nodes = len(self.x)
        else:
            assert n_nodes is not None, 'if no file is given, n_nodes is required'
            self.n_nodes = n_nodes
            self.instance_name = ''
            self.x, self.adj = u_i.make_instance(self.n_nodes, seed=self.seed)

    def get_features(self):
        return self.x, self.adj

    def check_solution(self, sol):

        assert len(sol) == len(self.x) + 1, 'len(sol) = ' + str(len(sol)) + ', n_nodes+1 = ' + str(len(self.x) + 1)
        assert len(sol) == len(set(sol)) + 1
        self.sim_counter += 1
        self.name = f'tour{self.sim_counter:03}'
        tour_time, rewards, pen, feas = u_o.tour_check(sol, self.x, self.adj, self.maxT_pen,
                                                       self.tw_pen, self.n_nodes)
        return tour_time, rewards, pen, feas


if __name__ == '__main__':
    env = Env(n_nodes=5, seed=1235)
    sol = [1, 2, 1, 4, 3, 5]
    print(sol)
    for _ in range(10):
        print(env.check_solution(sol))