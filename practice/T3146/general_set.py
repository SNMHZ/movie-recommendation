import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

SEED = 777
VERBOSE = False

print("Load Movielens dataset")
# Load Data
DATA_DIR = '/opt/ml/input/data/train'
GENERAL_DIR = os.path.join(DATA_DIR, 'general')
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
print("Loaded data with shape: {}".format(raw_data.shape))


def generate_general_train_test_set(test_plays: pd.DataFrame, n_all=10, n_seq=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(SEED)
    trains, labels = [], []
    for usr_id, tp in test_plays.groupby('user', as_index=False):
        _n_all = min(tp.shape[0]//4, n_all)
        _n_seq = min(_n_all, n_seq)
        _n_static = _n_all - _n_seq
        _n_all = _n_static + _n_seq

        _idxs = np.random.permutation(tp.shape[0]-_n_seq)[:_n_static]
        _mask = tp.index.isin(tp.index[_idxs])
        for i in range(_n_seq):
            _mask[-i-1] = True
        if VERBOSE:
         if _n_all != 10:
            print('_n_all:', _n_all)
            print(usr_id, _idxs)
            print(_n_static, _n_seq)

        trains.append(tp[~_mask])
        labels.append(tp[_mask])
        
    train_df = pd.concat(trains)
    label_df = pd.concat(labels)
    return train_df, label_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_all', type=int, default=10)
    parser.add_argument('--n_seq', type=int, default=2)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    SEED = args.seed
    train_df, label_df = generate_general_train_test_set(raw_data, args.n_all, args.n_seq)
    print('train_df:', train_df.shape)
    print('label_df:', label_df.shape)
    
    train_df.to_csv(os.path.join(GENERAL_DIR, 'train_ratings.csv'), index=False)
    label_df.to_csv(os.path.join(GENERAL_DIR, 'test_ratings.csv'), index=False)
    print('Done')