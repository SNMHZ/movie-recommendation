import os

import pandas as pd
import numpy as np

import random
from tqdm import tqdm

seed=42
random.seed(seed)
np.random.seed(seed)

def load_dataset(data_path='/opt/ml/input/data/train/'):
    return pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))


class SplitDataset:
    def random_sampling(n: int =5): 
        """
        n: default 5
        """
        rating = load_dataset()
        total_user_id = rating['user'].unique()
        total_user_num = rating.shape[0]
        val_user_idx = []
        train_user_idx = []
        print('start sampling...')
        for i in tqdm(total_user_id):
            idx = rating[rating['user'] == i].sample(n, random_state=seed).index
            val_user_idx.extend(idx)
        train_user_idx.extend(set(range(total_user_num)) - set(val_user_idx))

        train_set = rating.iloc[train_user_idx]
        val_set = rating.iloc[val_user_idx]

        train_set = train_set.reset_index(drop=True)
        val_set = val_set.reset_index(drop=True)

        print('Sampling Done.')
        return train_set, val_set