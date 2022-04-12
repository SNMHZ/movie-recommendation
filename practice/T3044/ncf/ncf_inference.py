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


### inference
with open('mdoel_iter5_1024_256_64_lr001_freezeitem', 'rb') as f:
    model = pickle.load(f)    


raw_rating_df['rating'] = -100
inference_matrix = raw_rating_df.pivot_table('rating', 'user', 'item').fillna(0)

inference_data = implicit_df[implicit_df.implicit_feedback != 1]
inference_X = inference_data.loc[:, implicit_df.columns != 'implicit_feedback']
inference_y = inference_data['implicit_feedback']

dataset = TensorDataset(torch.LongTensor(np.array(inference_X)), torch.FloatTensor(np.array(inference_y)))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = model.to(device)
model.eval()
for batch, (x, y) in enumerate(tqdm(dataloader, total=len(dataloader), bar_format="{l_bar}{r_bar}")):
    with torch.no_grad():
        in_x = x.to(device)
        output = model(in_x)

        x = x.numpy()
        for u, i, r in zip(x[:,0], x[:,1], output.to('cpu').detach().numpy()):
            inference_matrix.loc[user_dict[u], movie_dict[i]] = r


result = np.argpartition(inference_matrix, -10).iloc[:, -10:]
final_users, final_items = list(), list()
item_columns = inference_matrix.columns
for idx in range(result.shape[0]):
    final_users.extend([result.index[idx]] * 10)
    for i in result.values[idx]:
        final_items.append(item_columns[i])
        
submission_df = pd.DataFrame(zip(final_users,final_items), columns=['user','item'])
submission_df.to_csv("./ncf_iter5_1024_256_64_lr001_freezeitem.csv", index=False)