# Script configuration:
source = "data/processed/ratings_20k_users_train.csv"
destination = "models/demo_svd_pca.pkl"

# Allow import to reach for app module:
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports
import pandas as pd
from surprise import SVD, accuracy, Dataset, Reader, NormalPredictor, BaselineOnly
from src import pca, analyze_axes
from src.latent_factors import item_factors_df, user_factors_df
from datetime import datetime
import pickle

# Model:
notes = "SVD+PCA(n_factors=50, n_epochs=40, pca=7, lr=0.007, reg=0.06, random_state=100)"

# Train and export model:
df = pd.read_csv(source)
reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=50, n_epochs=40, lr_all=0.007, reg_all=0.06, random_state=100)
algo.fit(trainset)
algo = pca.PcaSvd(algo, n_components=7)

idf = item_factors_df(algo, trainset)
artifact = {
    "trainset": trainset,
    "created_at": datetime.now(),
    "dataset": source,
    "notes": notes,
    "users": user_factors_df(algo, trainset),
    "items": idf,
    "axes": analyze_axes.tag_correlations_df(idf)
}

with open(destination, "wb") as f:
    pickle.dump(artifact, f)
