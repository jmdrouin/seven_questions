# Allow import to reach for app module:
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports
import pandas as pd
from surprise import SVD, accuracy, Dataset, Reader, NormalPredictor, BaselineOnly
from surprise.model_selection import KFold
import numpy as np
from src import pca
from datetime import datetime
import json
import src.metrics as metrics

# Config:
source = "data/processed/ratings_20k_users_train.csv"
destination = "models/explore_models_results.txt"

def make_test_plan():
    # Baseline predictors for comparison:
    test_plan = [
        {"algo": NormalPredictor(), "desc": "random"},
        {"algo": BaselineOnly(), "desc": "baseline"}
    ]
    return test_plan

    # Test n_factors between 40 and 60
    for k in range(41,60):
        test_plan.append({
            "algo": SVD(n_factors=k, n_epochs=20, random_state=100),
            "desc": "SVD e=20 f="+str(k)
        })

    # Test wider range of n_factors
    for k in [10,20,30,40,60,70,80,90,100]:
        test_plan.append({
            "algo": SVD(n_factors=k, n_epochs=20, random_state=100),
            "desc": "SVD e=20 f="+str(k)
        })

    # Test effect of pca
    for k in [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50]:
        test_plan.append({
            "algo": SVD(n_factors=50, n_epochs=40, lr_all=0.007, reg_all=0.06, random_state=100),
            "pca": k,
            "desc": "SVD e=40 f=50 pca="+str(k)
        })

    # Test different parameters
    for n_epochs in [10, 20, 30, 40, 50]:
        for learning_rate in [0.004, 0.007, 0.001]:
            for reg_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
                test_plan_4.append({
                    "algo": SVD(n_factors=50, n_epochs=n_epochs, lr_all=learning_rate, reg_all=reg_factor, random_state=100),
                    "desc": f"SVD f=50 e={n_epochs} lr={learning_rate} reg={reg_factor})"
                })

    return test_plan    

def explore_models():
    kf = KFold(n_splits=3, random_state=100, shuffle=True)
    df = pd.read_csv(source)
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)
    test_plan = make_test_plan()

    for item in test_plan:
        desc = item["desc"]
        result = {
            "algo": desc,
            "ndcg_at_10": [],
            "rmse": [],
            "fcp": []
        }
        for trainset, testset in kf.split(data):
            algo = item["algo"]
            algo.fit(trainset)

            if "pca" in item:
                n_pca_components = item["pca"]
                algo = pca.PcaSvd(algo, n_components=n_pca_components)
                preds = pca.pca_predict(algo, trainset, testset)
            else:
                preds = algo.test(testset)
            
            result["ndcg_at_10"].append(metrics.ndcg(preds, k=10))
            result["rmse"].append(metrics.rmse(preds))
            result["fcp"].append(accuracy.fcp(preds, verbose=False))
        print(datetime.now())

        for metric in ["ndcg_at_10", "rmse", "fcp"]:
            result[metric + "_mean"] = np.mean(result[metric])
        print(result)

        with open(destination, "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

if __name__ == "__main__":
    explore_models()