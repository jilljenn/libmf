import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score, roc_auc_score
from collections import Counter
import sys


# df_train = pd.read_csv('/home/jj/Documents/Mangaki/mugen_train.csv', sep=' ', names=('user', 'item', 'v'))
# df = pd.read_csv('/home/jj/Documents/Mangaki/mugen_test.csv', sep=' ', names=('user', 'item', 'v'))
# df_train = pd.read_csv('/home/jj/code/cfdr/code/gottlieb_train.csv', sep=' ', names=('user', 'item', 'v'))
# df = pd.read_csv('/home/jj/code/cfdr/code/gottlieb_test.csv', sep=' ', names=('user', 'item', 'v'))
df_train = pd.read_csv('all_one_matrix.tr.txt', sep=' ', names=('user', 'item', 'v'))
df = pd.read_csv('all_one_matrix.te.txt', sep=' ', names=('user', 'item', 'v'))
df['user_id'] = np.unique(df['user'], return_inverse=True)[1]
df['item_id'] = np.unique(df['item'], return_inverse=True)[1]
encode_user = dict(df[['user', 'user_id']].drop_duplicates().to_numpy())
encode_item = dict(df[['item', 'item_id']].drop_duplicates().to_numpy())
df_train['user_id'] = df_train['user'].map(encode_user)
df_train['item_id'] = df_train['item'].map(encode_item)

n_users = df['user'].nunique()
n_items = df['item'].nunique()
n_entries, _ = df.shape
print(n_users, df['user_id'].max())
print(n_items, df['item_id'].max())
truth = csr_matrix((df['v'] * n_entries, (df['user_id'], df['item_id'])), shape=(n_users, n_items))

for filename in sys.argv[1:]:
    out = pd.read_csv(filename, names=('pred',))
    print(filename, out.shape, df['user_id'].shape)
    sparse = csr_matrix((out['pred'], (df['user_id'], df['item_id'])), shape=(n_users, n_items))
    
    ndcg_values = []
    ndcg10_values = []
    auc_values = []
    for user_id in df['user_id'].unique():
        test_set = list(set(range(n_items)) - set(df_train.query("user_id == @user_id")['item_id'].tolist()))
        # print(n_items - len(test_set))
        user_truth = truth[user_id, test_set].toarray()
        # print(Counter(user_truth.reshape(-1).tolist()))
        user_pred = sparse[user_id, test_set].toarray()
        ndcg_values.append(ndcg_score(user_truth, user_pred))
        ndcg10_values.append(ndcg_score(user_truth, user_pred, k=10))
        try:
            auc_values.append(roc_auc_score(user_truth.reshape(-1),
                                        user_pred.reshape(-1)))
        except ValueError as e:
            print(e)
            print(df.query("user_id == @user_id").shape)
            print(df.query("user_id == @user_id and v == 1"))
            print(df_train.query("user_id == @user_id"))
            print(user_id, len(test_set), Counter(user_truth.tolist()[0]))
            break
            continue
    print(filename)
    print('ndcg =', np.mean(ndcg_values))
    print('ndcg@10 =', np.mean(ndcg10_values))
    print('auc =', np.mean(auc_values))
