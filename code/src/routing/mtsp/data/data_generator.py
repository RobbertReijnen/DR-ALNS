import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import os
import pickle


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def generate_mtsp_data(dataset_size, mtsp_size):
    return np.random.uniform(size=(dataset_size, mtsp_size+1, 2)).tolist()


if __name__ == "__main__":
    dataset = generate_mtsp_data(dataset_size=10000, mtsp_size=100)
    file = 'mtsp_100_10000.pkl'
    if os.path.isfile(file):
        print('file already exists')
    else:
        save_dataset(dataset, file)
