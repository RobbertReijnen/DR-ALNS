import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import gymnasium as gym
import random
from alns import ALNS
import numpy as np
import numpy.random as rnd
from pathlib import Path

from routing.tsp.alns_tsp import tsp_helper_functions

from routing.tsp.alns_tsp.tsp_env import tspEnv
from routing.tsp.alns_tsp.destroy_operators import neighbor_graph_removal, random_removal, relatedness_removal
from routing.tsp.alns_tsp.repair_operators import regret_insertion
from routing.tsp.alns_tsp.initial_solution import compute_initial_solution


class tspAlnsEnv_LSA1(gym.Env):
    def __init__(self, config, **kwargs):

        # Parameters
        self.config = config["environment"]
        self.rnd_state = rnd.RandomState()

        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # LOAD INSTANCE
        base_path = Path(__file__).resolve().parents[2]
        self.instance_file = str(base_path.joinpath(self.config["instance_file"]))

        self.instances = self.config["instance_nr"]
        self.instance = None

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        # Gym-related part
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = self.config["iterations"]  # max number of generations in an episode

        # Action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([3, 1, 10, 100])
        self.observation_space = gym.spaces.Box(shape=(8,), low=0, high=100, dtype=np.float64)

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
             self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
            dtype=np.float64).squeeze()

        return state

    def reset(self, seed=None, options=None):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        SEED = random.randint(0, 100000)

        # randomly select problem instance
        self.instance = random.choice(self.instances)
        nb_customers, dist_matrix_data = tsp_helper_functions.read_input_tsp(self.instance_file, self.instance)

        random_state = rnd.RandomState(SEED)
        state = tspEnv([], nb_customers, dist_matrix_data, self.instance_file, SEED)

        self.initial_solution = compute_initial_solution(state, random_state)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state)
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)

        self.dr_alns.add_repair_operator(regret_insertion)

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

        factors = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0}
        nr_nodes_to_remove = round(factors[action[2]] * current.nb_customers)

        self.temperature = (1/(action[3]+1)) * self.max_temperature


        if nr_nodes_to_remove == current.nb_customers:
            nr_nodes_to_remove -= 1

        destroyed = d_operator(current, self.rnd_state, nr_nodes_to_remove)

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
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1

        elif new_current != current and new_current.objective() <= current.objective():
            self.current_solution = new_current
            self.current_updated = 1

        if new_current.objective() > current.objective():
            self.improvement = 1

        self.cost_difference_from_best = (self.current_solution.objective() / self.best_solution.objective()) * 100

        # update graph of current and best solutions
        self.current_solution.graph = self.best_solution.graph = tsp_helper_functions.update_neighbor_graph(candidate, candidate.route, candidate.objective())

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
        for episode in range(2):
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

    env = tspAlnsEnv_LSA1(get_parameters("tspAlnsEnv_LSA1"))

    print('Start training')
    model = Trainer("tspAlnsEnv_LSA1", "models").create_model()
    model.train()

