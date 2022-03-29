import dataloader
import model
import world
import utils
import torch
import numpy as np
import pandas as pd
import os
dataset = dataloader.MovieInferenceLoader()
Recmodel = model.LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
try:
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    world.cprint(f"loaded model weights from {weight_file}")
except FileNotFoundError:
    print(f"{weight_file} not exists, start from beginning")

u_batch_size = world.config['test_u_batch_size']

testDict = dataset.testDict
# eval mode with no dropout
Recmodel = Recmodel.eval()
max_K = max(world.topks)
with torch.no_grad():
    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
    # auc_record = []
    # ratings = []
    total_batch = len(users) // u_batch_size + 1
    for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to(world.device)
        rating = Recmodel.getUsersRating(batch_users_gpu)
        #rating = rating.cpu()
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        # aucs = [ 
        #         utils.AUC(rating[i],
        #                   dataset, 
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    path = '../../data/train'
    data = pd.read_csv(os.path.join(path,'train_ratings.csv'))
    itemids = data['item'].unique()
    de_users = data['user'].unique().repeat(10)
    en_items = []
    for t_batch in rating_list:
        en_items.extend(np.array(t_batch).flatten().tolist())
    de_items = [itemids[i] for i in en_items]
    inference = pd.DataFrame(zip(de_users,de_items), columns = ['user','item'])
    inference.to_csv('lightGCN_submission.csv', index=False)