import numpy as np
import pandas as pd
import os
path = '../../data/train'
data = pd.read_csv(os.path.join(path,'train_ratings.csv'))
users = data['user'].unique()
items = data['item'].unique()
user2id = dict((uid, i) for (i, uid) in enumerate(users))
item2id = dict((sid, i) for (i, sid) in enumerate(items))
encoded_user = [user2id[u] for u in data['user']]
encoded_item = [item2id[i] for i in data['item']]
full_df = pd.DataFrame(zip(encoded_user,encoded_item), columns=['user','item'])
N = full_df.shape[0]
t_N = int(N*0.2)
test_idx = np.random.choice(range(N), t_N, replace=False)
train_idx = np.setdiff1d(range(N), test_idx)
train = full_df.iloc[train_idx]
test = full_df.iloc[test_idx]
full_df.to_csv("../../data/train/light_gcn/full.csv", index=False)
train.to_csv("../../data/train/light_gcn/train.csv", index=False)
test.to_csv("../../data/train/light_gcn/test.csv", index=False)