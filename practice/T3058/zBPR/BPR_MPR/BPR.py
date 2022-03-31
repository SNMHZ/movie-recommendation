# Implement BPR.
# Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback.
# Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI, 2009. 
# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
# import scores

class BPR:
    user_count = 31360
    item_count = 6807
    latent_factors = 100
    lr = 1e-4
    reg = 0.01
    train_count = 10
    train_data_path = '/opt/ml/zBPR/BPR_MPR/train.txt'
    test_data_path = '/opt/ml/zBPR/BPR_MPR/test.txt'
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def preprocessing(self):
        ratings = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
        self.user_dict = {user:i for i,user in enumerate(set(ratings["user"]))}
        self.idx2user_dict = {i:user for i,user in enumerate(set(ratings["user"]))}
        self.item_dict = {item:i for i,item in enumerate(set(ratings["item"]))}
        
    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = self.user_dict[int(u)]
                i = self.item_dict[int(i)]
                user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = self.user_dict[int(line[0])]
            item = self.item_dict[int(line[1])]
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            # u = random.randint(1, self.user_count)
            u=user
            if u not in user_ratings_train.keys():
                continue
            # sample a positive item from the observed items
            # i = random.sample(user_ratings_train[u], 1)[0]
            for i in user_ratings_train[u]:
            # sample a negative item from the unobserved items
                for j in range(self.item_count):
            # j = random.randint(1, self.item_count)
            # while j in user_ratings_train[u]:
            #     j = random.randint(1, self.item_count)
                    if j in user_ratings_train[u]:
                        continue
                    r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
                    r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
                    r_uij = r_ui - r_uj
                    loss_func = -1.0 / (1 + np.exp(r_uij))
                    # update U and V
                    self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
                    self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
                    self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
                    # update biasV
                    self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
                    self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def main(self):
        self.preprocessing()
        self.user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in tqdm(range(self.user_count), desc = "user_count"):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # training
        for i in tqdm(range(self.train_count), desc = "train_count"):
            self.train(self.user_ratings_train)
        self.predict_matrix = self.predict(self.U, self.V).getA()
        # prediction
        self.predict_matrix = pre_handel(self.user_ratings_train, self.predict_matrix, self.item_count)
        self.predict_ = self.predict_matrix.reshape(-1)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)

        # Top-K evaluation
        # scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)
        topK_df = np.argpartition(self.predict_matrix, -10)[:,-10:]
        # topK_df = self.predict_matrix[np.arange(self.predict_matrix.shape[0])[:, None],top]

        first = True
        for i, item_list in tqdm(enumerate(topK_df), desc = "submit"):
            for j in item_list:
                tmp_df = pd.DataFrame({"user":bpr.idx2user_dict[i],"item":j}, index = [0])
                if first:
                    first = False
                    return_df = tmp_df
                else:
                    return_df = pd.concat([return_df, tmp_df], axis = 0, sort = False)
        return_df = return_df.sort_values(by=["user"])
        return_df.to_csv("/opt/ml/output/submission.csv", index = False)
        # return_df.to_csv("/opt/ml/output/submission.csv", index = False)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[u][j] = 0
    
    return predict

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()
