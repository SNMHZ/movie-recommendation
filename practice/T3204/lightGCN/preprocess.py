import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def random_split(df):
    global train_df
    global test_df
    t_r, t_e = train_test_split(df, test_size=0.2)
    train_df = pd.concat([train_df, t_r])
    test_df = pd.concat([test_df, t_e])
    

path = '../../data/train'
data = pd.read_csv(os.path.join(path,'train_ratings.csv'))
users = data['user'].unique()
items = data['item'].unique()
user2id = dict((uid, i) for (i, uid) in enumerate(users))
item2id = dict((sid, i) for (i, sid) in enumerate(items))
encoded_user = [user2id[u] for u in data['user']]
encoded_item = [item2id[i] for i in data['item']]
full_df = pd.DataFrame(zip(encoded_user,encoded_item), columns=['user','item'])
train_df = pd.DataFrame({'user' : [], 'item' : []}, columns = ['user','item'])
test_df = pd.DataFrame({'user' : [], 'item' : []}, columns = ['user','item'])
full_df.groupby('user').apply(random_split)
train_df = train_df.astype('int')
test_df = test_df.astype('int')
full_df.to_csv("../../data/train/light_gcn/full.csv", index=False)
train_df.to_csv("../../data/train/light_gcn/train.csv", index=False)
test_df.to_csv("../../data/train/light_gcn/test.csv", index=False)