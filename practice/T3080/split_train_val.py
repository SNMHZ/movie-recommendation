import os

import pandas as pd
import numpy as np

import random
from tqdm import tqdm

seed = 42

random.seed(seed)
np.random.seed(seed)

train_data_dir = '/opt/ml/input/data/train/'

train_rating = pd.read_csv(os.path.join(train_data_dir, 'train_ratings.csv'))

total_user_id = train_rating['user'].unique()

val_user_idx = []
for i in tqdm(total_user_id):
    idx = train_rating[train_rating['user'] == i].sample(5, random_state=seed).index
    val_user_idx.extend(idx)

val_set = train_rating.iloc[val_user_idx]

train_idx = set(range(train_rating.shape[0])) - set(val_user_idx)
train_idx = list(train_idx)

train_set = train_rating.iloc[train_idx]

train_set = train_set.reset_index(drop=True)
val_set = val_set.reset_index(drop=True)

train_set.to_csv('/opt/ml/train_set.csv', index=False)
val_set.to_csv('/opt/ml/val_set.csv', index=False)