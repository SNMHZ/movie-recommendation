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

# 1. Rating df 생성
rating_data = "/opt/ml/input/data/train/train_ratings.csv"

raw_rating_df = pd.read_csv(rating_data)
raw_rating_df
raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)
print("Raw rating df")
print(raw_rating_df)

users = set(raw_rating_df.loc[:, 'user'])
items = set(raw_rating_df.loc[:, 'item'])

#2. Genre df 생성
genre_data = "/opt/ml/input/data/train/genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop 
# print(raw_genre_df)

genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경
print("Raw genre df - changed to id")
print(raw_genre_df)

# 3. Negative instance 생성
print("Create Nagetive instances")
num_negative = 1000
user_group_dfs = list(raw_rating_df.groupby('user')['item'])
first_row = True
user_neg_dfs = pd.DataFrame()

for u, u_items in user_group_dfs:
    u_items = set(u_items)
    i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)  # 관측 없는 데이터 중에서 num만큼 추출
    
    i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})  # 0점짜리 negative df 생성
    if first_row == True:
        user_neg_dfs = i_user_neg_df
        first_row = False
    else:
        user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)  # 행으로 concat

raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)  # concat(positive, negative)

# 4. Join dfs
joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner') 
# print("Joined rating df")
# print(joined_rating_df)

# 5. user, item을 zero-based index로 mapping
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
print("Data")
print(data)

n_data = len(data)
n_user = len(users)
n_item = len(items)
n_genre = len(genres)

print("# of data : {}\n# of users : {}\n# of items : {}\n# of genres : {}".format(n_data, n_user, n_item, n_genre))

#6. feature matrix X, label tensor y 생성
user_col = torch.tensor(data.loc[:,'user'])  
item_col = torch.tensor(data.loc[:,'item'])  
genre_col = torch.tensor(data.loc[:,'genre'])

offsets = [0, n_user, n_user+n_item]  # [0, 31360, 38167]
for col, offset in zip([user_col, item_col, genre_col], offsets): # [u, i, g] + [0, n_user, n_user + n_item]
    col += offset

X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)
y = torch.tensor(list(data.loc[:,'rating']))


#7. data loader 생성
class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)


dataset = RatingDataset(X, y)
train_ratio = 0.9

train_size = int(train_ratio * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
all_data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 8. model
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
        return y

device = torch.device('cuda')
input_dims = [n_user, n_item, n_genre]
embedding_dim = 200
model = DeepFM(input_dims, embedding_dim, mlp_dims=[30, 20, 10]).to(device)
bce_loss = nn.BCELoss() # Binary Cross Entropy loss
lr, num_epochs = 0.001, 150
optimizer = optim.Adam(model.parameters(), lr=lr)

print('training start.')

for e in range(num_epochs) :
    print(f'epoch: {e}')
    for x, y in all_data_loader:
        x, y = x.to(device), y.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = bce_loss(output, y.float())
        loss.backward()
        optimizer.step()
        
# save model
MODEL_PATH = '/opt/ml/input/code/experiment/deep_fm'
torch.save(model.state_dict(), 
           os.path.join(MODEL_PATH, "deepFM_neg1000_emb200_iter150_statedict.pt")) 

print('training done. ')