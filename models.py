"""
models.py
Item-based collaborative filtering and SVD-based latent factor recommender.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import joblib

class ItemCF:
    def __init__(self):
        self.item_sim = None
        self.item_index_to_id = None

    def fit(self, R_csr):
        """
        Fit item-item similarity using cosine similarity on item vectors (items x users).
        R_csr: user x item sparse matrix
        """
        # compute item vectors (transpose to item x user)
        item_user = R_csr.T.tocsr()
        # cosine_similarity yields dense matrix; keep memory in mind (ml-100k small)
        self.item_sim = cosine_similarity(item_user)
        print("[ItemCF] item_sim shape:", self.item_sim.shape)

    def recommend_for_user(self, user_row, R_csr, top_k=10, exclude_rated=True):
        """
        user_row: integer user index (row in R_csr)
        R_csr: user x item matrix
        returns top_k item indices (column indices)
        """
        # user vector
        user_vec = R_csr[user_row].toarray().ravel()  # ratings for user
        # score items = sum(similarities * user_ratings)
        scores = self.item_sim.dot(user_vec)
        if exclude_rated:
            rated = (user_vec > 0)
            scores[rated] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0

    def fit(self, R_csr):
        """
        Fit TruncatedSVD on the user-item matrix (fill missing as zeros).
        R_csr: user x item sparse matrix
        """
        print("[SVDRecommender] Fitting TruncatedSVD...")
        # convert to dense? Instead we use TruncatedSVD on sparse matrix directly
        svd = TruncatedSVD(n_components=min(self.n_components, min(R_csr.shape)-1), random_state=42)
        # TruncatedSVD acts on user-item sparse matrix -> returns user_factors (transformed)
        user_factors = svd.fit_transform(R_csr)  # shape (n_users, n_components)
        item_factors = svd.components_.T  # shape (n_items, n_components)
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.global_mean = R_csr.data.mean() if R_csr.data.size > 0 else 0.0
        print("[SVDRecommender] Done. user_factors shape:", self.user_factors.shape)

    def predict_user_item(self, user_idx, item_idx):
        # dot product of user and item factors (approx rating)
        return float(self.user_factors[user_idx].dot(self.item_factors[item_idx]))

    def recommend_for_user(self, user_idx, R_csr, top_k=10, exclude_rated=True):
        user_vec = R_csr[user_idx].toarray().ravel()
        scores = self.user_factors[user_idx].dot(self.item_factors.T)
        if exclude_rated:
            rated = (user_vec > 0)
            scores[rated] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"[models] Model saved to {path}")

def load_model(path):
    return joblib.load(path)
