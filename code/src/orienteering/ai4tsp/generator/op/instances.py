import numpy as np
import os
import pandas as pd
import random
from os import path

from orienteering.ai4tsp.generator.op.prizes import PrizeGenerator
from orienteering.ai4tsp.generator.op.timewindows import TWGenerator


class InstanceGenerator:
    ''''This is the instance generator
    In this class we generate the instances for each DOPTW
    If given a seed it will always generate the same travel times
    '''

    base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

    def __init__(self, n_instances, n_nodes, w=(20, 40, 60, 80, 100), ylim=(0, 50), xlim=(0, 200), prize='distance',
                 seed=None, data_dir='data/test', offset=0):
        # seed the generation process
        self.seed = seed
        np.random.seed(seed=seed)
        random.seed(seed)

        self.n_instances = n_instances
        self.n_nodes = n_nodes
        self.w = w
        self.xlim = xlim
        self.ylim = ylim
        self.prize = prize
        self.offset = offset
        self.data_dir = path.join(self.base_dir, data_dir)

        assert prize in ('distance', 'constant', 'uniform'), 'prize has to be: distance, constant or uniform'

        self.tw_gen = TWGenerator()
        self.op_gen = PrizeGenerator()

    def generate_coord(self):

        coord = np.zeros((self.n_nodes, 2))
        coord[:, 0] = np.random.randint(low=self.xlim[0], high=self.xlim[1], size=self.n_nodes).astype(float)
        coord[:, 1] = np.random.randint(low=self.ylim[0], high=self.ylim[1], size=self.n_nodes).astype(float)

        return coord

    def generate_instance_files(self, save=True):

        indices = np.arange(1, self.n_nodes + 1)
        blanks = np.zeros(self.n_nodes)
        inst_optw, adj = None, None
        for i in range(self.n_instances):
            coord = self.generate_coord()
            data = {'CUSTNO': indices, 'XCOORD': coord[:, 0], 'YCOORD': coord[:, 1], 'DEMAND': blanks,
                    'READY TIME': blanks, 'DUE DATE': blanks, 'SERVICE TIME': blanks}

            inst_df = pd.DataFrame(data)
            inst_df = inst_df.set_index('CUSTNO')
            # get the time windows
            inst_tw, adj = self.tw_gen(inst_df, w=np.random.choice(self.w))
            # get the prizes and maximum length
            inst_optw = self.op_gen(inst_tw, adj, self.prize)
            if save:
                self.save_optw(i, inst_optw, adj)

        return inst_optw, adj

    def make_dir(self, name):

        path_ = os.path.join(self.data_dir, name)
        if os.path.exists(path_):
            pass
        else:
            os.makedirs(path_)
        return path_

    def save_optw(self, i, inst, adj, x_dir='instances', adj_dir='adjs'):
        path_x = self.make_dir(x_dir)
        path_adj = self.make_dir(adj_dir)

        inst.to_csv(os.path.join(path_x, f'instance{self.offset + i + 1:04}.csv'), index=False, sep=',')
        adj.to_csv(os.path.join(path_adj, f'adj-instance{self.offset + i + 1:04}.csv'), index=False, sep=',')


if __name__ == '__main__':
    gen = InstanceGenerator(250, n_nodes=200, offset=750)
    gen.generate_instance_files()