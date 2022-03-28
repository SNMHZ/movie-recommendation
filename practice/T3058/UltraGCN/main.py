from IPython import embed
import torch
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import argparse
from model import UltraGCN
from utils import pload,pstore
from train import train
from inference import inference

def data_param_prepare(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}
    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
    
 
    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda')
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight


    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos
    
    

    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']
    submission_file_path = config['Submission']['submission_file_path']

    # dataset processing
    train_data, test_data, submission_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, test_file_path, submission_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)
    submission_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num


    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    
    # Compute \Omega to extend UltraGCN to the item-item occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)

    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, submission_loader, mask, test_ground_truth_list, interacted_items


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

    

def load_data(train_file, test_file, submission_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    submissionUniqueUsers, submissionItem, submissionUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize, submissionDataSize = 0, 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    with open(submission_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                submissionUniqueUsers.append(uid)
                submissionUser.extend([uid] * len(items))
                submissionItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                submissionDataSize += len(items)

    train_data = []
    test_data = []
    submission_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    for i in range(len(submissionUser)):
        submission_data.append([submissionUser[i], submissionUser[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0


    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, submission_data, train_mat, n_user, m_item, constraint_mat






# The below 7 functions (hit, ndcg, RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) follow this license.
# MIT License

# Copyright (c) 2020 Xiang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################### TESTING #####################################




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="/opt/ml/git/level2-movie-recommendation-level2-recsys-13/practice/T3058/UltraGCN/movie_rec.ini", type=str, help='config file path')
    #이부분은 default값을 바꾸시거나 python인자를 넣어주세요 debug를 위해 바꿨습니다. ./movie_rec.ini

    args = parser.parse_args()

    print('###################### UltraGCN ######################')


    print('1. Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, submission_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)


    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    inference(ultragcn, submission_loader, mask, params['topk'])
    print('END')
