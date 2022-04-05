import os
import time
import argparse
import pandas as pd
from multiprocessing import Pool

DATA_DIR = '/opt/ml/input/data/train'
GENERAL_DIR = os.path.join(DATA_DIR, 'general')

try:
    label_df = pd.read_csv(os.path.join(GENERAL_DIR, 'test_ratings.csv'), header=0)
except:
    print('No test_ratings.csv found')
    exit(0)

def _worker_getRecall(user_df):
        user, submission_df = user_df
        preds = label_df[label_df['user'] == user]['item']
        labels = submission_df[submission_df['user'] == user]['item']

        return preds.isin(labels).sum() / labels.shape[0]

def getRecall(submission_df):
    with Pool(os.cpu_count()) as p:
        users = label_df['user'].unique()
        result = p.map(_worker_getRecall, zip(users, [submission_df]*len(users)) )
    return sum(result) / label_df['user'].nunique()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='g_submission.csv')
    args = parser.parse_args()

    print('Load submission file')
    start = int(time.time())
    submission_df = pd.read_csv(args.file, header=0)
    print('Loaded submission file with shape: {}'.format(submission_df.shape))

    print('Get recall')
    print('recall:', getRecall(submission_df))

    print("***run time(sec) :", int(time.time()) - start)