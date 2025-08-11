"""
evaluate.py
Evaluate SVD RMSE and ItemCF precision@k using holdout test.
"""

import argparse
import numpy as np
import pandas as pd
from preprocess import load_ratings, build_user_item_matrix, train_test_split_leave_one_out
from models import ItemCF, SVDRecommender
from sklearn.metrics import mean_squared_error

def evaluate_svd(train_df, test_df, n_components=50):
    R_train, user_map, item_map, inv_user, inv_item = build_user_item_matrix(train_df)
    svd = SVDRecommender(n_components=n_components)
    svd.fit(R_train)
    # predict for test pairs
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        u = row['user_id']; i = row['item_id']; r = row['rating']
        if u not in user_map or i not in item_map:
            continue
        uidx = user_map[u]; iidx = item_map[i]
        y_true.append(r)
        y_pred.append(svd.predict_user_item(uidx, iidx))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[evaluate_svd] Test RMSE: {rmse:.4f} (n_test={len(y_true)})")
    return rmse

def precision_at_k(itemcf, R_train, test_df, user_map, item_map, k=10):
    correct = 0
    total = 0
    for _, row in test_df.iterrows():
        u = row['user_id']; i = row['item_id']
        if u not in user_map or i not in item_map:
            continue
        uidx = user_map[u]; iidx = item_map[i]
        top_idx, _ = itemcf.recommend_for_user(uidx, R_train, top_k=k, exclude_rated=True)
        if iidx in top_idx:
            correct += 1
        total += 1
    prec = correct / total if total > 0 else 0.0
    print(f"[precision_at_k] Precision@{k}: {prec:.4f} ({correct}/{total})")
    return prec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ratings.csv')
    args = parser.parse_args()
    df = load_ratings(args.data)
    train_df, test_df = train_test_split_leave_one_out(df)
    R_train, user_map, item_map, inv_user, inv_item = build_user_item_matrix(train_df)
    # SVD RMSE
    evaluate_svd(train_df, test_df, n_components=50)
    # ItemCF precision@10
    itemcf = ItemCF()
    itemcf.fit(R_train)
    precision_at_k(itemcf, R_train, test_df, user_map, item_map, k=10)

if __name__ == "__main__":
    main()
