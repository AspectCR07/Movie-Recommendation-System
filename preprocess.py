"""
preprocess.py
Load ratings.csv and create train/test splits and sparse matrices for modeling.
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def load_ratings(path="data/ratings.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found: {path}. Run data_fetcher.build_ratings_csv()")
    df = pd.read_csv(path)
    return df

def build_user_item_matrix(df):
    """
    Returns:
      - R: csr_matrix shape (n_users, n_items)
      - user_map: dict user_id -> row idx
      - item_map: dict item_id -> col idx
      - inv_user_map, inv_item_map: inverse maps
    """
    users = np.sort(df['user_id'].unique())
    items = np.sort(df['item_id'].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: j for j, it in enumerate(items)}
    rows = df['user_id'].map(user_map)
    cols = df['item_id'].map(item_map)
    data = df['rating'].values
    R = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    inv_user = {v: k for k, v in user_map.items()}
    inv_item = {v: k for k, v in item_map.items()}
    return R, user_map, item_map, inv_user, inv_item

def train_test_split_leave_one_out(df, random_state=42):
    """
    For each user, hold out one rating for test (if multiple ratings exist).
    Returns train_df and test_df.
    """
    test_rows = []
    train_rows = []
    grouped = df.groupby('user_id')
    rng = np.random.RandomState(random_state)
    for user, group in grouped:
        if len(group) == 1:
            # keep single rating in train (no test for that user)
            train_rows.append(group.index[0])
            continue
        # choose one index as test
        idx = rng.choice(group.index)
        test_rows.append(idx)
        train_rows.extend([i for i in group.index if i != idx])
    train_df = df.loc[train_rows].reset_index(drop=True)
    test_df = df.loc[test_rows].reset_index(drop=True)
    return train_df, test_df

if __name__ == "__main__":
    df = load_ratings()
    tr, te = train_test_split_leave_one_out(df)
    print("Train size:", len(tr), "Test size:", len(te))
