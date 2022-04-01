import csv
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

##### setting #####
# Rating df 생성
rating_data = "/opt/ml/input/data/train/train_ratings.csv"

raw_rating_df = pd.read_csv(rating_data)
raw_rating_df
raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

users = set(raw_rating_df.loc[:, 'user'])
items = set(raw_rating_df.loc[:, 'item'])

# Genre df 생성
genre_data = "/opt/ml/input/data/train/genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop 

genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경

# train용 Negative instance 생성
print("Create Nagetive instances")
num_negative = 50
user_group_dfs = list(raw_rating_df.groupby('user')['item'])
first_row = True
user_neg_dfs = pd.DataFrame()

for u, u_items in tqdm(user_group_dfs):
    u_items = set(u_items)
    i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)  # 관측 없는 데이터 중에서 num만큼 추출
    
    i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})  # 0점짜리 negative df 생성
    if first_row == True:
        user_neg_dfs = i_user_neg_df
        first_row = False
    else:
        user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)  # 행으로 concat

raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)  # concat(positive, negative) 
joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner') 

# user, item을 zero-based index로 mapping
users = list(set(joined_rating_df.loc[:,'user']))  # len = 31360
users.sort()
items =  list(set(joined_rating_df.loc[:, 'item']))  # len = 6807
items.sort()
genres =  list(set(joined_rating_df.loc[:, 'genre']))  # 0 ~ 17
genres.sort()

if len(users)-1 != max(users):  # -> index 작업 안 되어 있으면, 다시 매핑
    users_dict = {users[i]: i for i in range(len(users))}
    joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
    users = list(set(joined_rating_df.loc[:,'user']))
    
if len(items)-1 != max(items):
    items_dict = {items[i]: i for i in range(len(items))}
    joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
    items =  list(set((joined_rating_df.loc[:, 'item'])))

joined_rating_df = joined_rating_df.sort_values(by=['user'])
joined_rating_df.reset_index(drop=True, inplace=True)

data = joined_rating_df

n_data = len(data)
n_user = len(users)
n_item = len(items)
n_genre = len(genres)

print("creating negative instances done")

# model
class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_genre

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)
        
        self.embedding = nn.Embedding(total_input_dim, embedding_dim) 
        self.embedding_dim = len(input_dims) * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) #TODO 1 : linear layer를 넣어주세요.
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         #TODO 2 : torch.sum을 이용하여 square_of_sum을 작성해주세요(hint : equation (2))
        sum_of_square = torch.sum(embed_x ** 2, dim=1)         #TODO 3 : torch.sum을 이용하여 sum_of_square을 작성해주세요(hint : equation (2))
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        embed_x = self.embedding(x)
        
        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        embed_x = self.embedding(x)
        #fm component
        fm_y = self.fm(x).squeeze(1)
        
        #deep component
        mlp_y = self.mlp(x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        return y, fm_y + mlp_y


##### inference #####
## data 준비
# joined_all_df
with open('/opt/ml/input/code/experiment/deep_fm/user_all_neg_dfs', 'rb') as f:
    user_all_neg_dfs = pickle.load(f)

joined_all_df = pd.merge(user_all_neg_dfs, raw_genre_df, on='item', how='left') 
joined_all_df

# zero-based index로 mapping
joined_all_df['user']  = joined_all_df['user'].map(lambda x : users_dict[x])
joined_all_df['item']  = joined_all_df['item'].map(lambda x : items_dict[x])

inference_data = joined_all_df.sort_values(by=['user'])
inference_data.reset_index(drop=True, inplace=True)

# col 로드
with open('user_col', 'rb') as f:
    user_col = pickle.load(f)
with open('item_col', 'rb') as f:
    item_col = pickle.load(f)
with open('genre_col', 'rb') as f:
    genre_col = pickle.load(f)

offsets = [0, n_user, n_user+n_item]  # [0, 31360, 38167]
for col, offset in zip([user_col, item_col, genre_col], offsets): # [u, i, g] + [0, n_user, n_user + n_item]
    col += offset

# dataset, data loader 생성
X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
y = torch.tensor(list(inference_data.loc[:,'rating']))  # 사용 x

class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)


inference_dataset = RatingDataset(X, y)
inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=True)

## model 준비
# 모델 불러오기
device = torch.device('cuda')
input_dims = [n_user, n_item, n_genre]
embedding_dim = 200
model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
MODEL_PATH = '/opt/ml/input/code/experiment/deep_fm'
model.load_state_dict(torch.load(os.path.join(
    MODEL_PATH, "deepFM_neg200_emb200_iter150_statedict.pt")))

## inference
print('inference started.')
# make u*i matrix
base_path = '/opt/ml/input/data/train'
train_df_path = os.path.join(base_path, 'train_ratings.csv')

train_raing_df = pd.read_csv(train_df_path)
train_raing_df['viewed'] = -100
user_item_matrix = train_raing_df.pivot_table('viewed', 'user', 'item').fillna(0)

dict_genre = dict(map(reversed, genre_dict.items()))
dict_items = dict(map(reversed, items_dict.items()))
dict_users = dict(map(reversed, users_dict.items()))

# make matrix
for x, y in inference_loader:
    model.eval()
    in_x = x.to(device)
    output = model(in_x)[1]
    result = torch.round(output[1])
    
    x = x.numpy() - offsets
    for u, i, r in zip(x[:,0], x[:,1], output.to('cpu').detach().numpy()):
        user_item_matrix.loc[dict_users[u], dict_items[i]] = r

# make csv
result = np.argpartition(user_item_matrix, -10).iloc[:, -10:]
final_users, final_items = list(), list()
item_columns = user_item_matrix.columns
for idx in range(result.shape[0]):
    final_users.extend([result.index[idx]] * 10)
    for i in result.values[idx]:
        final_items.append(item_columns[i])
        
submission_df = pd.DataFrame(zip(final_users,final_items), columns=['user','item'])
submission_df.to_csv("/opt/ml/input/code/experiment/deep_fm/neg1000_emb200_iter150.csv", index=False)

print('inference done.')