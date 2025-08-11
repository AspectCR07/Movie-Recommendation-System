"""
data_fetcher.py
Downloads MovieLens 100k dataset (if not present) and prepares a ratings CSV.
"""

import os
import zipfile
import requests
from io import BytesIO
import pandas as pd

ML100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_ml100k(dest_folder="data"):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "ml-100k.zip")
    extracted_flag = os.path.join(dest_folder, "ml-100k", "u.data")
    if os.path.exists(extracted_flag):
        print("[data_fetcher] ml-100k already present, skipping download.")
        return os.path.join(dest_folder, "ml-100k")
    print("[data_fetcher] Downloading MovieLens 100k dataset (this may take a bit)...")
    r = requests.get(ML100K_URL, stream=True, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(dest_folder)
    print("[data_fetcher] Extracted to", dest_folder)
    return os.path.join(dest_folder, "ml-100k")

def build_ratings_csv(dest_folder="data"):
    ml_dir = download_ml100k(dest_folder)
    udata = os.path.join(ml_dir, "u.data")
    # u.data format: user id | item id | rating | timestamp
    df = pd.read_csv(udata, sep='\t', names=["user_id", "item_id", "rating", "ts"])
    ratings_csv = os.path.join(dest_folder, "ratings.csv")
    df.to_csv(ratings_csv, index=False)
    print(f"[data_fetcher] ratings.csv saved to {ratings_csv}")
    return ratings_csv

if __name__ == "__main__":
    build_ratings_csv("data")
