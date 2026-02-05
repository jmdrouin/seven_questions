import pandas as pd
from surprise import SVD, accuracy, Dataset, Reader, NormalPredictor, BaselineOnly
from surprise.model_selection import KFold
import numpy as np
import pickle
from sklearn.metrics import ndcg_score, root_mean_squared_error
from collections import defaultdict
from src.smodel import pca, analyze_axes
from datetime import datetime
import json

def ndcg(preds, k=10, score_transform=None):
    if score_transform is None: score_transform = lambda x: x

    by_user = defaultdict(list)
    for p in preds:
        by_user[p.uid].append(p)

    scores = []
    for uid, ps in by_user.items():
        true_values = np.array([p.r_ui for p in ps], dtype=float).reshape(1, -1)
        pred_values  = np.array([p.est  for p in ps], dtype=float).reshape(1, -1)
        score = ndcg_score(
            score_transform(true_values),
            score_transform(pred_values),
            k = min(k, true_values.shape[1])
        )
        scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")

def rmse(preds, score_transform=lambda x: x):
    y_true = [p.r_ui for p in preds]
    y_pred = [p.est for p in preds]
    return root_mean_squared_error(score_transform(y_true), score_transform(y_pred))

def cloglog(r, lower_bound=0, upper_bound=5.5):
    zero_to_one_rating = (r - lower_bound) / (upper_bound - lower_bound)
    true_cloglog = np.log(-np.log(1 - zero_to_one_rating))
    return true_cloglog

def inv_cloglog(y, lower_bound=0, upper_bound=5.5):
    z = 1 - np.exp(-np.exp(y))
    x = lower_bound + z * (upper_bound - lower_bound)
    return x

def item_factors_df(algo, trainset):
    q = algo.qi
    b = algo.bi
    factors = []
    for inner_iid in range(trainset.n_items):
        raw_iid = trainset.to_raw_iid(inner_iid)
        vec = q[inner_iid]
        bias = b[inner_iid]
        factors.append([raw_iid, bias] + list(vec))

    num_q_cols = len(factors[0]) - 2
    q_cols = ["q" + str(i) for i in range(num_q_cols)]
    return pd.DataFrame(factors, columns=["movieId", "bias"] + q_cols)

def user_factors_df(algo, trainset):
    p = algo.pu
    b = algo.bu
    factors = []
    for inner_uid in range(trainset.n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        vec = p[inner_uid]
        bias = b[inner_uid]
        factors.append([raw_uid, bias] + list(vec))

    num_p_cols = len(factors[0]) - 2
    p_cols = ["p" + str(i) for i in range(num_p_cols)]
    return pd.DataFrame(factors, columns=["userId", "bias"] + p_cols)

def explore_models():
    kf = KFold(n_splits=3, random_state=100, shuffle=True)

    transform = None #"cloglog"
    df = pd.read_csv("data/processed/ratings_20k_users_train.csv")

    if transform == "cloglog":
        df["rating"] = cloglog(df["rating"])

    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

    test_plan_1 = [
        {"algo": NormalPredictor(), "desc": "random", "transform": transform},
        {"algo": BaselineOnly(), "desc": "baseline", "transform": transform}
    ]
    for k in [50,10,20,30,40,60,70,80,90,100]:
        test_plan_1.append({
            "algo": SVD(n_factors=k, n_epochs=20, random_state=100),
            "desc": "SVD e=20 f="+str(k),
            "transform": transform
        })

    test_plan_2 = []
    for k in range(41,60):
        test_plan_2.append({
            "algo": SVD(n_factors=k, n_epochs=20, random_state=100),
            "desc": "SVD e=20 f="+str(k),
            "transform": transform
        })

    test_plan_3 = []
    for k in [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50]:
        test_plan_3.append({
            "algo": SVD(n_factors=50, n_epochs=40, lr_all=0.007, reg_all=0.06, random_state=100),
            "pca": k,
            "desc": "SVD e=40 f=50 pca="+str(k),
            "transform": transform
        })

    test_plan_4 = []
    for n_epochs in [10, 20, 30, 40, 50]:
        for learning_rate in [0.004, 0.007, 0.001]:
            for reg_factor in [0.01, 0.02, 0.03]:
                test_plan_4.append({
                    "algo": SVD(n_factors=50, n_epochs=n_epochs, lr_all=learning_rate, reg_all=reg_factor, random_state=100),
                    "desc": f"SVD f=50 e={n_epochs} lr={learning_rate} reg={reg_factor})",
                    "transform": transform
                })

    test_plan_5 = []
    for n_epochs in [30, 40, 50]:
        for learning_rate in [0.004, 0.007]:
            for reg_factor in [0.06, 0.07, 0.08]:
                test_plan_5.append({
                    "algo": SVD(n_factors=50, n_epochs=n_epochs, lr_all=learning_rate, reg_all=reg_factor, random_state=100),
                    "desc": f"SVD f=50 e={n_epochs} lr={learning_rate} reg={reg_factor})",
                    "transform": transform
                })

    test_plan = test_plan_3

    for item in test_plan:
        desc = item["desc"]
        if transform == "cloglog":
            desc = desc + " t=cloglog"
        print(desc)
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
            
            score_transform = lambda x: x
            if item["transform"] == "cloglog":
                score_transform = inv_cloglog

            result["ndcg_at_10"].append(ndcg(preds, k=10, score_transform=score_transform))
            result["rmse"].append(rmse(preds, score_transform=score_transform))
            result["fcp"].append(accuracy.fcp(preds, verbose=False))
        print(datetime.now())

        for metric in ["ndcg_at_10", "rmse", "fcp"]:
            result[metric + "_mean"] = np.mean(result[metric])
        print(result)

        with open("explore_models_results.txt", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

def make_demo_model():
    df = pd.read_csv("data/processed/ratings_20k_users_train.csv")
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=40, lr_all=0.007, reg_all=0.06, random_state=100)
    svd.fit(trainset)
    algo = pca.PcaSvd(svd, n_components=7)


    idf = item_factors_df(algo, trainset)
    artifact = {
        "trainset": trainset,
        "created_at": datetime.now(),
        "dataset": "ratings_normal_users_small",
        "notes": "SVD(n_factors=50, n_epochs = 20, pca = 7)",
        "users": prepare_model.user_factors_df(algo, trainset),
        "items": idf,
        "axes": analyze_axes.tag_correlations_df(idf)
    }

    with open("models/demo_svd_pca_dump.pkl", "wb") as f:
        pickle.dump(artifact, f)

if __name__ == "__main__":
    #explore_models()
    make_demo_model()