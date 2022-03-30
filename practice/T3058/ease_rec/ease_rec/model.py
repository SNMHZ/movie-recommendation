from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user'])
        items = self.item_enc.fit_transform(df.loc[:, 'item'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        df = pd.DataFrame()
        items = set(self.item_enc.transform(items))
        dd = train.loc[train.user.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item)
        dd['cu'] = self.user_enc.transform(dd.user)
        g = dd.groupby('user')
        for user, group in tqdm(g):
            watched = items - set(group['ci'])
            candidates = list(watched)
            u = group['cu'].iloc[0]
            pred = np.take(self.pred[u, :], candidates)
            res = np.argpartition(pred, -k)[-k:]
            r = pd.DataFrame({
                "user": [user] * len(res),
                "item": np.take(candidates, res),
                "score": np.take(pred, res)
            }).sort_values('score', ascending=False)
            df = pd.concat([df,r], axis=0, ignore_index=True)
        df['item'] = self.item_enc.inverse_transform(df['item'])
        return df
