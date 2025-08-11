"""
utils.py
Utility helpers - kept minimal for now.
"""

def ensure_data_present(path="data/ratings.csv"):
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python data_fetcher.py` or use --download flag.")
