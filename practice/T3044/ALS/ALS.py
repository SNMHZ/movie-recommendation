from typing import Union, Tuple, List

import os
import pandas as pd
import seaborn as sns
import scipy
import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
from IPython.display import Image
import warnings
import pickle

tqdm.pandas()
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)

# 데이터 불러오기
base_path = '/opt/ml/input/data/train'
train_df_path = os.path.join(base_path, 'train_ratings.csv')
df = pd.read_csv(train_df_path)
df['viewed'] = np.ones(df.shape[0])

# MF 모델의 파라미터(p_u, q_i)를 업데이트 하는 ALS 함수
def als(
    F: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    K: int,
    regularization: float
) -> None:
    """
    MF 모델의 파라미터를 업데이트하는 ALS

    :param F: (np.ndarray) 유저-아이템 preference 매트릭스. shape: (유저 수, 아이템 수)
    :param P: (np.ndarray) 유저의 잠재 요인 행렬. shape: (유저 수, 잠재 요인 수)
    :param Q: (np.ndarray) 아이템의 잠재 요인 행렬. shape: (아이템 수, 잠재 요인 수)
    :param C: (np.ndarray) 평점 테이블에 Confidence Level을 적용한 행렬. 
               shape: (유저 수, 아이템 수)
    :param K: (int) 잠재 요인 수
    :param regularization: (float) l2 정규화 파라미터
    :return: None
    """
    for user_id, F_user in enumerate(F):
        C_u = np.diag(C[user_id])
        left = np.linalg.inv(np.matmul(np.matmul(Q.T, C_u), Q) + regularization * np.eye(K))
        right = np.matmul(np.matmul(Q.T, C_u), F_user)
        P[user_id] = np.dot(left, right)
        
        
    for item_id, F_item in enumerate(F.T):
        C_i = np.diag(C[:, item_id])
        left = np.linalg.inv(np.matmul(np.matmul(P.T, C_i), P) + regularization * np.eye(K))
        right = np.matmul(np.matmul(P.T, C_i), F_item)
        Q[item_id] = np.dot(left, right)

# ALS의 Loss를 계산하는 함수
def get_ALS_loss(
    F: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    regularization: float
) -> float:
    """
    전체 학습 데이터(실제 평가를 내린 데이터)에 대한 ALS의 Loss를 계산합니다.
    
    :param F: (np.ndarray) 유저-아이템 preference 매트릭스. shape: (유저 수, 아이템 수)
    :param P: (np.ndarray) 유저의 잠재 요인 행렬. shape: (유저 수, 잠재 요인 수)
    :param Q: (np.ndarray) 아이템의 잠재 요인 행렬. shape: (아이템 수, 잠재 요인 수)
    :param C: (np.ndarray) 평점 테이블에 Confidence Level을 적용한 행렬. shape: (유저 수, 아이템 수)
    :param regularization: (float) l2 정규화 파라미터
    :return: (float) 전체 학습 데이터에 대한 Loss
    """
    
    user_index, item_index = F.nonzero()
    loss = 0
    for user_id, item_id in zip(user_index, item_index):
        predict_error = pow(F[user_id, item_id] - np.dot(P[user_id].T, Q[item_id]), 2)
        confidence_error = C[user_id, item_id] * predict_error
        loss += confidence_error
    for user_id in range(F.shape[0]):
        regularization_term = regularization * np.sum(np.square(P[user_id]))
        loss += regularization_term
    for item_id in range(F.shape[1]):
        regularization_term = regularization * np.sum(np.square(Q[item_id]))
        loss += regularization_term

    return loss
    
# ALS 기반 MF Train

class MF_ALS(object):
    
    def __init__(self, F, K, C, regularization, epochs, verbose=False):
        self.F = F
        self.num_users, self.num_items = F.shape
        self.K = K
        self.C = C
        self.regularization = regularization
        self.epochs = epochs
        self.verbose = verbose
        
        self.training_process = list()
    
    def train(self):
        
        # 유저, 아이템 잠재 요인 행렬 초기화
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        for epoch in range(1, self.epochs + 1):
            als(self.F, self.P, self.Q, self.C, self.K, self.regularization)
            loss = get_ALS_loss(self.F, self.P, self.Q, self.C, self.regularization)
            self.training_process.append((epoch, loss))
            if self.verbose and (epoch % 1 == 0):
                print("epoch: %d, error = %.4f" % (epoch, loss))
        
        self.training_process = pd.DataFrame(self.training_process, columns = ['epoch', 'rmse'])
    
    def get_predicted_full_matrix(self):
        return np.matmul(self.P, self.Q.T)

# input 행렬
user_item_matrix = df.pivot_table('viewed', 'user', 'item').fillna(0)
user_num = user_item_matrix.shape[0]
movie_num = user_item_matrix.shape[1]
preference_matrix = np.copy(user_item_matrix.iloc[:user_num,:movie_num])
preference_matrix[preference_matrix > 0] = 1

# 하이퍼파라미터
alpha = 40  # confidence lavel, 논문에서 추천
C = 1 + alpha * np.copy(preference_matrix)
K = 200  # dimension of latent ventor, 논문에서 추천
regularization = 40  # r_lambda, 논문에서 추천
epochs = 15
verbose = True  # 학습 과정의 status print 옵션

# train
print('train start !!')
mf_als = MF_ALS(preference_matrix, K, C, regularization, epochs, verbose)
mf_als.train()
print('train done !!')

# 유저*행렬 매트릭스(결과 값)
als_model_df = pd.DataFrame(np.matmul(mf_als.P, mf_als.Q.T), columns=user_item_matrix.iloc[:user_num,:movie_num].columns, index=user_item_matrix.iloc[:user_num,:movie_num].index)

# 값 저장(pickle)
with open('als_model_iter15', 'wb') as f:
    pickle.dump(als_model_df, f)

# top 10 선정
als_model_df_final = als_model_df - preference_matrix * 1000
result = np.argpartition(als_model_df_final, -10).iloc[:, -10:]

# submission 생성
users, items = list(), list()
item_columns = als_model_df_final.columns
for idx in range(result.shape[0]):
    users.extend([result.index[idx]] * 10)
    for i in result.values[idx]:
        items.append(item_columns[i])
        
test_df = pd.DataFrame(zip(users,items), columns=['user','item'])
test_df.to_csv("/opt/ml/input/code/experiment/als/als_submission.csv", index=False)


