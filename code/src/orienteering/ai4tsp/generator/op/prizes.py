import math
import numpy as np
import pandas as pd

import orienteering.ai4tsp.op_utils.heuristics as h
import orienteering.ai4tsp.op_utils.op as u


class PrizeGenerator:

    def __call__(self, ins_tw, D, prize='distance', depot=1):
        """
        receives a dataframe with 4 columns and n rows where n is the number of nodes.
        the columns are [ 'XCOORD.', 'YCOORD.', 'TW_LOW', 'TW_HIGH'] showing the x and y coordinates
        and left and right bound of time window.
        this function adds a new columns named PRIZE that contains the prize for each node.
        the type of the prize is an input and it could be constant, uniform or distance
        """

        # it is missing the maximum length of the op tour, otherwise the problem is just a max prize
        # This parameter can be computed from half the optimal uncostrained tsp tour - need to call concorde here

        n_nodes = len(ins_tw.index)

        # create a nearest neighbour tour
        nn_tour = h.nn_algo(depot, D, n_nodes)

        # sanity check
        assert len(set(nn_tour)) == n_nodes
        assert len(nn_tour) == n_nodes + 1

        # max distance to the depot
        d_max = np.max(D.loc[1, :])

        # minT is d_max*2 because we have to guarantee we can go back and forth from the largest possible distance
        minT = d_max * 2

        # maxT needs to be larger than minT but not too large otherwise the problem is too easy
        maxT = u.tsp_tour_cost(nn_tour, D)
        # since a nn tour is unlikely to be tsp optimal we discount the cost by 0.5
        # this should not be too hard as the travel times are max travel times
        maxT = max(minT * 2, math.ceil(maxT * 0.5))

        # finally selecting T to be between minT and maxT - we can be conservative and generate only on maxT
        T = np.random.randint(minT, maxT)
        # T = maxT

        T_ = np.repeat(T, n_nodes)

        # generating prizes
        if prize == 'constant':
            p = np.ones(n_nodes)
            p[0] = 0
        elif prize == 'uniform':
            p = np.random.randint(1, 101, n_nodes)
            p = p / 100
            p[0] = 0
        elif prize == 'distance':
            p = np.zeros(n_nodes)
            for j in range(1, n_nodes):
                p[j] = 1 + math.floor(99 * D.loc[1, j + 1] / d_max)
            p = p / 100

        indices = np.arange(1, n_nodes + 1)
        ins_twop = ins_tw
        ins_twop['PRIZE'] = pd.Series(p, index=indices)
        ins_twop['MAXTIME'] = pd.Series(T_, index=indices)

        return ins_twop