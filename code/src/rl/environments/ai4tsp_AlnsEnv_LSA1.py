import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import gymnasium as gym
import random
from alns import ALNS
import numpy as np
import numpy.random as rnd
from pathlib import Path

from orienteering.ai4tsp.alns_ai4tsp import ai4tsp_helper_functions

from orienteering.ai4tsp.alns_ai4tsp.ai4tsp_env import ai4tspEnv
from orienteering.ai4tsp.alns_ai4tsp.destroy_operators import random_removal, relatedness_removal, neighbor_graph_removal
from orienteering.ai4tsp.alns_ai4tsp.repair_operators import random_best_distance_repair, random_best_prize_repair, random_best_ratio_repair, regret_insertion
from orienteering.ai4tsp.alns_ai4tsp.initial_solution import empty_route


# training with AI4TSP problem
base_path = Path(__file__).resolve().parents[2]
test_data_instance_path = base_path.joinpath('orienteering/ai4tsp/data/test/instances')
test_data_adj_path = base_path.joinpath('orienteering/ai4tsp/data/test/adjs')


class ai4tspAlnsEnv_LSA1(gym.Env):
    def __init__(self, config, **kwargs):

        # Parameters
        self.config = config["environment"]
        self.rnd_state = rnd.RandomState()

        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # Instances configuration
        self.instances = self.config["instances"]
        self.instance = random.choice(self.instances)

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        self.best_routes = []

        # Gym-related part
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = self.config["iterations"]  # max number of generations in an episode

        # Action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 10, 100])
        self.observation_space = gym.spaces.Box(shape=(7,), low=0, high=100, dtype=np.float64)

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if -self.current_solution.objective() == -self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [self.improvement, self.cost_difference_from_best, is_current_best,
             self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
            dtype=np.float64).squeeze()

        return state

    def reset(self, seed=None, options=None):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        SEED = random.randint(0, 100000000)
        random_state = rnd.RandomState(SEED)

        # randomly select problem instance
        self.instance = random.choice(self.instances)

        # Load instance and create initial solution
        x_path = os.path.join(test_data_instance_path, self.instance + '.csv')
        adj_path = os.path.join(test_data_adj_path, 'adj-' + self.instance + '.csv')
        x, adj, instance_name = ai4tsp_helper_functions.read_instance(x_path, adj_path)

        nodes = [(i + 1) for i in range(0, len(x))]

        random_state = rnd.RandomState()
        state = ai4tspEnv(nodes, [], x, adj, instance_name, SEED)
        self.current_solution = empty_route(state, init_node=1)
        self.initial_solution = copy.deepcopy(self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state)
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)

        self.dr_alns.add_repair_operator(random_best_distance_repair)
        self.dr_alns.add_repair_operator(random_best_prize_repair)
        self.dr_alns.add_repair_operator(random_best_ratio_repair)

        # reset tracking values
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0

        self.iteration, self.reward = 0, 0
        self.done = False

        return self.make_observation(), {}

    def step(self, action, **kwargs):
        self.iteration += 1
        self.stagcount += 1
        self.current_updated = 0
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0
        self.current_improved = 0

        current = self.current_solution
        best = self.best_solution

        d_idx, r_idx = action[0], action[1]
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx]

        self.temperature = (1/(action[3]+1)) * self.max_temperature

        factors = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0}
        destroyed = d_operator(current, self.rnd_state, degree_of_destruction=factors[action[2]])

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state)

        new_best, new_current = self.consider_candidate(best, current, candidate)

        if new_best != best and new_best is not None:
            # found new best solution
            self.best_solution = new_best
            self.current_solution = new_best
            self.current_updated = 1
            self.reward += 5
            self.stagcount = 0
            self.current_improved = 1

        elif new_current != current and new_current.objective() > current.objective():
            # solution accepted, because better than current, but not better than best
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1

        elif new_current != current and new_current.objective() <= current.objective():
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1

        if -new_current.objective() > -current.objective():
            self.improvement = 1

        a = self.current_solution.objective()
        b= self.best_solution.objective()
        if abs(a) == 0 or abs(b) == 0:
            self.cost_difference_from_best = -1
        else:
            self.cost_difference_from_best = (-a / -b) * 100

        self.current_solution.graph = self.best_solution.graph = ai4tsp_helper_functions.update_neighbor_graph(candidate, candidate.route, candidate.objective())
        state = self.make_observation()

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True

        return state, self.reward, self.done, False, {}

    # --------------------------------------------------------------------------------------------------------------------

    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
        probability = np.exp((curr.objective() - cand.objective()) / self.temperature)

        # best:
        if cand.objective() < best.objective():
            return cand, cand

        # accepted:
        elif probability >= rnd.random():
            return None, cand

        else:
            return None, curr

    # --------------------------------------------------------------------------------------------------------------------

    def run(self, model, episodes=1):
        """
        Use a trained model to select actions
        """
        try:
            for episode in range(episodes):
                self.done = False
                state = self.reset()
                while not self.done:
                    action = model.predict(state)
                    state, reward, self.done, _ = self.step(action[0])
                    # print(state, reward, self.iteration)
        except KeyboardInterrupt:
            pass

    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(1000):
            self.done = False
            state = self.reset()
            print("start episode: ", episode, " with start state: ", state)
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print(
                    "step {}, action: {}, New state: {}, Reward: {:2.3f}".format(
                        self.iteration, action, state, reward
                    )
                )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from rl.baselines import get_parameters, Trainer

    env = ai4tspAlnsEnv_LSA1(get_parameters("ai4tspAlnsEnv_LSA1"))

    print('Start training')
    model = Trainer("ai4tspAlnsEnv_LSA1", "models").create_model()
    model.train()


