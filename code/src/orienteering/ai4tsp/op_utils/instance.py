import pandas as pd
from pathlib import Path

from orienteering.ai4tsp.generator.op.instances import InstanceGenerator


def make_instance(n_nodes, seed=None, save=False):
    generator = InstanceGenerator(1, n_nodes, seed=seed)
    x_df, adj_df = generator.generate_instance_files(save)
    x, adj = x_df.to_numpy(), adj_df.to_numpy()
    return x, adj


def read_instance(x_path, adj_path):
    x_df = pd.read_csv(x_path, sep=',')
    adj_df = pd.read_csv(adj_path, sep=',')

    x, adj = x_df.to_numpy(), adj_df.to_numpy()

    return x, adj, Path(x_path).stem