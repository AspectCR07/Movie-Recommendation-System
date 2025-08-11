"""
recommend.py
Usage examples:
  python recommend.py --method itemcf --user 10 --data data/ratings.csv --topk 10
  python recommend.py --method svd --user 10 --data data/ratings.csv --topk 10
"""

import argparse
import numpy as np
import pandas as pd
from data_fetcher import build_ratings_csv
from preprocess import load_ratings, build_user_item_matrix, train_test_split_leave_one_out
from models import ItemCF, SVDRecommender
import os

def train_models(train_df):
    R_train, user_map, item_map, inv_user, inv_item = build_user_item_matrix(train_df)
    itemcf = ItemCF()
    itemcf.fit(R_train)
    svd = SVDRecommender(n_components=50)
    svd.fit(R_train)
    return {"itemcf": (itemcf, R_train, user_map, item_map, inv_user, inv_item),
            "svd": (svd, R_train, user_map, item_map, inv_user, inv_item)}

def recommend_for_user_id(user_id, method_obj, R, user_map, item_map, inv_item, topk=10):
    if user_id not in user_map:
        raise KeyError(f"User id {user_id} not found in training data.")
    uidx = user_map[user_id]
    model = method_obj
    top_idx, scores = model.recommend_for_user(uidx, R, top_k=topk, exclude_rated=True)
    item_ids = [inv_item[i] for i in top_idx]
    return item_ids, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['itemcf','svd'], default='itemcf')
    parser.add_argument('--user', type=int, required=True, help='Original MovieLens user id (1..943)')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--data', default='data/ratings.csv')
    parser.add_argument('--download', action='store_true', help='Download dataset first')
    args = parser.parse_args()

    if args.download:
        build_ratings_csv("data")

    df = load_ratings(args.data)
    train_df, test_df = train_test_split_leave_one_out(df)
    models = train_models(train_df)
    model_obj, R_train, user_map, item_map, inv_user, inv_item = models[args.method]

    rec_items, scores = recommend_for_user_id(args.user, model_obj, R_train, user_map, item_map, inv_item, topk=args.topk)
    print(f"\nTop-{args.topk} recommendations for user {args.user} (method={args.method}):")
    for rank, (iid, sc) in enumerate(zip(rec_items, scores), start=1):
        print(f"{rank:2d}. MovieID {iid}  score={sc:.4f}")

if __name__ == "__main__":
    main()
