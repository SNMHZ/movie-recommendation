### library
import csv
import os
import pickle
import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import normal_
from torch.utils.data import DataLoader, Dataset, TensorDataset

seed = 42
random.seed(seed)
np.random.seed(seed)


### 데이터
## 데이터 준비
rating_data = "/opt/ml/input/data/train/train_ratings.csv"
raw_rating_df = pd.read_csv(rating_data)
raw_rating_df
raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

user_ids = raw_rating_df['user'].unique()
movie_ids = raw_rating_df['item'].unique()

ratings_matrix = raw_rating_df.pivot(index='user', columns='item', values='rating')

''' 데이터 로드 '''
with open('implicit_df', 'rb') as f:
    implicit_df = pickle.load(f)
with open('user_dict', 'rb') as f:
    user_dict = pickle.load(f)
with open('movie_dict', 'rb') as f:
    movie_dict = pickle.load(f)

# 효율성을 위해 category타입으로 변경
implicit_df['user'] = implicit_df['user'].astype("category")
implicit_df['item'] = implicit_df['item'].astype("category")

## 데이터 분리
train_X = implicit_df.loc[:, implicit_df.columns != 'implicit_feedback']
train_y = implicit_df['implicit_feedback']

## 데이터 셋
dataset = TensorDataset(torch.LongTensor(np.array(train_X)), torch.FloatTensor(np.array(train_y)))


### 모델
class MLPLayers(nn.Module):
    """
    여러 층의 MLP Layer Class
    
    Args:
        - layers: (List) input layer, hidden layer, output layer의 node 수를 저장한 List.
                ex) [5, 4, 3, 2] -> input layer: 5 nodes, output layer: 2 nodes, hidden layers: 4 nodes, 3 nodes
        - dropout: (float) dropout 확률
    Shape:
        - Input: (torch.Tensor) input features. Shape: (batch size, # of input nodes)
        - Output: (torch.Tensor) output features. Shape: (batch size, # of output nodes)
    """
    def __init__(self, layers, dropout):
        super(MLPLayers, self).__init__()
        
        # initialize Class attributes
        self.layers = layers
        self.n_layers = len(self.layers) - 1
        self.dropout = dropout
        self.activation = nn.ReLU()
        
        # define layers
        mlp_modules = list()
        for i in range(self.n_layers):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            input_size = self.layers[i]
            output_size = self.layers[i+1]
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(self.activation)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        self.apply(self._init_weights)
        
    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class NCF(nn.Module):
    """
    Neural Collaborative Filtering
    
    Args:
        - n_users: (int) 전체 유저의 수
        - n_items: (int) 전체 아이템의 수
        - emb_dim: (int) Embedding의 Dimension
        - layers: (List) Neural CF Layers의 각 node 수를 저장한 List.
                ex) [5, 4, 3, 2] -> hidden layers: 5 nodes, 4 nodes, 3 nodes, 2 nodes
        - dropout: (float) dropout 확률
        - pretrained: (str) pretrained된 임베딩 weight 위치
    Shape:
        - Input: (torch.Tensor) input features, (user_id, item_id). Shape: (batch size, 2)
        - Output: (torch.Tensor) expected implicit feedback. Shape: (batch size,)
    """
    def __init__(self, n_users, n_items, emb_dim, layers, dropout, pretrained = None):
        super(NCF, self).__init__()
        
        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.layers = layers
        self.n_layers = len(self.layers) + 1
        self.dropout = dropout
        
        # define layers
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.mlp_layers = MLPLayers([2 * self.emb_dim] + self.layers, self.dropout)
        self.predict_layer = nn.Linear(self.layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)

        # load item_embedding's weight(pretrained)
        if pretrained is not None:
            with open(pretrained, 'rb') as f:
                pretrained_emb = pickle.load(f)
            pretrained_weight = pretrained_emb.weight[sorted(movie_dict.values()), :]
            
            item_weight = self.item_embedding.state_dict()
            item_weight['weight'] = pretrained_weight
            self.item_embedding.load_state_dict(item_weight)
        
    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, input_feature):
        user, item = torch.split(input_feature, [1, 1], -1)
        user = user.squeeze(-1)
        item = item.squeeze(-1)
        
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        
        input_feature = torch.cat((user_e, item_e), -1)
        mlp_output = self.mlp_layers(input_feature)
        output = self.predict_layer(mlp_output)
        output = self.sigmoid(output)
        return output.squeeze(-1)


### train
# 설정 및 하이퍼파라미터
batch_size = 2048
data_shuffle = True
emb_dim = 100
layers = [1024, 256, 64]
dropout = 0
epochs = 5
learning_rate = 0.001
gpu_idx = 0
early_stop = 5

n_users = raw_rating_df['user'].nunique()
n_items = raw_rating_df['item'].nunique()

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # 재현을 위한 설정
    torch.backends.cudnn.deterministic = True  # 재현을 위한 설정
device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=data_shuffle)
model = NCF(n_users, n_items, emb_dim, layers, dropout, pretrained='/opt/ml/input/code/experiment/pretrained_emb100').to(device)

loss_fn = nn.BCELoss().to(device)
err_fn = None
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# train
size = len(dataloader.dataset)
num_batches = len(dataloader)
save_loss = 99999
e_stop_stack = 0
print('training started.')
for e in range(epochs):
    if e_stop_stack > early_stop:
        break
    train_loss = 0
    print(f'Epoch {e+1} ...')
    for batch, (x, y) in enumerate(tqdm(dataloader, 
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}")):
        x, y = x.to(device), y.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # if (batch+1) % 1000 == 0:
        #     loss, current = loss.item(), batch * len(x)
        #     print(f"Loss: {loss:>7f} | [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    print(f'  - AVG Losses: {train_loss:>7f}')
    if train_loss < save_loss:
        e_stop_stack = 0
        save_loss = train_loss
        print(f'  - Better performance. Saving model ...')
        with open('mdoel_iter5_1024_256_64_lr001', 'wb') as f:
            pickle.dump(model, f)
    else:
        e_stop_stack += 0
    print()
print('training done.')

'''모델 로드'''
# with open('mdoel_iter5_1024_256_64_lr001', 'rb') as f:
    # model = pickle.load(f)