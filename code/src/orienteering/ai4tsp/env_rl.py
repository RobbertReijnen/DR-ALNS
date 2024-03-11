from orienteering.ai4tsp import env
import numpy as np
import orienteering.ai4tsp.op_utils.op as u_o


class EnvRL(env.Env):

    def __init__(self, n_nodes=None, seed=None, from_file=False, x_path=None, adj_path=None, verbose=False,
                 adaptive=True):
        super().__init__(n_nodes, seed, from_file, x_path, adj_path)
        self.sim_counter = 0
        self.verbose = verbose
        self.adaptive = adaptive

        self.current_node = None
        self.mask = None
        self.tour_time = None
        self.time_t = None

        self.feas = None
        self.return_to_depot = None
        self.rewards = None
        self.pen = None
        self.tw_high = None
        self.tw_low = None
        self.prizes = None
        self.maxT = None
        self.tour = None
        self.violation_t = None
        self.name = None

        self.reset()

    def get_seed(self):
        return self.seed

    def get_sim_name(self):
        return self.name

    def get_instance_name(self):
        return self.instance_name

    def visited(self, node):
        return bool(self.mask[node - 1])

    def check_solution(self, sol):
        if self.adaptive:
            pass
        else:
            # this is will generate a different randomness than 'step()'
            return u_o.tour_check(sol, self.x, self.adj, self.maxT_pen,
                                  self.tw_pen, self.n_nodes)

    def get_remaining_time(self):
        return self.maxT - self.tour_time

    def get_collected_rewards(self):
        return self.rewards

    def get_incurred_penalties(self):
        return self.pen

    def get_feasibility(self):
        return self.feas

    def get_current_violation(self):
        return self.violation_t

    def get_current_node(self):
        return self.current_node

    def is_tour_done(self):
        return self.return_to_depot

    def get_current_node_features(self):
        return self.x[self.current_node - 1]

    def _get_rewards(self, node):

        self.pen_t = 0
        self.rwd_t = 0
        self.violation_t = 0

        # only compute stuff if you are not back to depot
        if not self.return_to_depot:
            # make sure a node is not visited twice
            assert not self.visited(node), f'node: {node} already visited in the tour'
            assert node != 0, 'node: 0 (zero) is not allowed.'

            if self.tour_time > self.tw_high[node - 1]:
                self.feas = False
                # penalty added for each missed tw
                self.pen += self.tw_pen
                self.pen_t = self.tw_pen
                self.violation_t = 1

            elif self.tour_time < self.tw_low[node - 1]:
                # time added for being too early
                self.tour_time += self.tw_low[node - 1] - self.tour_time
                self.rewards += self.prizes[node - 1]
                self.rwd_t = self.prizes[node - 1]
            else:
                # within the time window - nothing to fix
                self.rewards += self.prizes[node - 1]
                self.rwd_t = self.prizes[node - 1]

            if node == 1:
                self.return_to_depot = True

            if self.tour_time > self.maxT:
                # penalty added for taking longer than maxT
                self.pen += self.maxT_pen * self.n_nodes
                self.pen_t += self.maxT_pen * self.n_nodes
                self.feas = False
                self.violation_t = 2

            # add the next node to the tour
            self.tour.append(node)
            self.mask[node - 1] = 1

    def step(self, node):

        if len(self.tour) >= self.n_nodes + 1:
            return None
        assert node <= self.n_nodes, f'node {node} does not exist for instance of size {self.n_nodes}'

        previous_tour_time = self.tour_time
        time = self.adj[self.current_node - 1, node - 1]
        noise = np.random.randint(1, 101, size=1)[0] / 100
        self.tour_time += np.round(noise * time, 2)
        self._get_rewards(node)
        self.time_t = self.tour_time - previous_tour_time
        self.current_node = node

        return self.tour_time, self.time_t, self.rwd_t, self.pen_t, self.feas, self.violation_t, self.return_to_depot

    def reset(self):

        self.current_node = 1
        self.mask = [0] * self.n_nodes
        self.tour_time = 0
        self.time_t = 0
        self.feas = True
        self.return_to_depot = False
        self.rewards = 0
        self.pen = 0
        self.tw_high = self.x[:, -3]
        self.tw_low = self.x[:, -4]
        self.prizes = self.x[:, -2]
        self.maxT = self.x[0, -1]
        self.tour = [self.current_node]
        self.violation_t = 0  # 0: none, 1: tw, 2: maxT (takes precedence on tw)
        self.sim_counter += 1
        self.name = f'tour{self.sim_counter:03}'
        if self.verbose:
            print(f'[*] Starting a new simulation: {self.name}')


if __name__ == '__main__':
    env = EnvRL(5, seed=19120623)
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)
    print(50 * '-')
    env.reset()
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)
