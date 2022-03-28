import torch
import pandas as pd
def inference(model, submission_loader, mask, topk):
    users_list = []
    rating_list = []
    user_dict = {}
    with open('/opt/ml/git/level2-movie-recommendation-level2-recsys-13/practice/T3058/UltraGCN/data/movie_rec/user_dict.txt', 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                user_dict[int(l[0])] = int(l[1])

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(submission_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

    
    i = 0
    first = True
    for tnsor in rating_list:
        for j in tnsor:
            tmp_df = pd.DataFrame({"user":user_dict[i],"item":j})
            i+=1
            if first:
                first = False
                submission_csv = tmp_df
            else:
                submission_csv = pd.concat([submission_csv, tmp_df], axis = 0, sort=False)
    submission_csv.to_csv("/opt/ml/output/submission.csv", index = False)
    