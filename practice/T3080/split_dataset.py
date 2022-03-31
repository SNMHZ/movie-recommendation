import os

import pandas as pd
import numpy as np

import random
from tqdm import tqdm

from sklearn.model_selection import train_test_split

seed=42
random.seed(seed)
np.random.seed(seed)

def load_dataset(data_path='/opt/ml/input/data/train/'):
    return pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))


class SplitDataset:
    def num_random_sampling(n: int =5): 
        """
        n: default 5
        """
        print('start sampling...')
        rating = load_dataset()
        total_user_id = rating['user'].unique()
        total_user_num = rating.shape[0]
        val_user_idx = []
        train_user_idx = []
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

    def ratio_random_sampling(r: float =0.1):
        """
        r: defalut 0.1
        minimum number of sample is 5
        """
        print('start sampling...')
        rating = load_dataset()
        df = rating['user'].value_counts() > round(5/r)
        idx = df[df].index
        rating[rating['user'].isin(idx)]['user'].value_counts()
        r_over_user_cnt = rating['user'].value_counts() > round(5/0.1)
        over_id_ls = r_over_user_cnt[r_over_user_cnt].index
        over_df = rating[rating['user'].isin(over_id_ls)]

        train_over_df, val_over_df = train_test_split(over_df, test_size=r, random_state=seed, stratify=over_df.user.values)
        under_idx = list(set(rating.index) - set(over_df.index))
        under_df = rating.iloc[under_idx].reset_index(drop=True)

        total_under_user_id = under_df['user'].unique()
        train_under_idx = []
        val_under_idx = []
        for i in total_under_user_id:
            idx = under_df[under_df['user'] == i].sample(5, random_state=42).index
            val_under_idx.extend(idx)
        train_under_idx.extend(set(under_df.index) - set(val_under_idx))

        train_under_df = under_df.iloc[train_under_idx]
        val_under_df = under_df.iloc[val_under_idx]

        train_set = pd.concat([train_over_df, train_under_df]).sort_values(by=['user','time']).reset_index(drop=True)
        val_set = pd.concat([val_over_df, val_under_df]).sort_values(by=['user','time']).reset_index(drop=True)

        print('Sampling Done.')
        return train_set, val_set