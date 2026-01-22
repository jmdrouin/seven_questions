import pandas as pd
from surprise import SVD, SVDpp, accuracy, Dataset, Reader, NormalPredictor, BaselineOnly
#from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import pickle

def make_demo_algo(trainset):
    algo = SVD(n_factors = 5, n_epochs = 25)
    algo.fit(trainset)

    from datetime import datetime
    artifact = {
        "model": algo,
        "trainset": trainset,
        "created_at": datetime.now(),
        "dataset": "ratings_normal_users_small",
        "notes": "SVD(n_factors=5, n_epochs = 25)",
        "users": user_factors_df(algo, trainset),
        "items": item_factors_df(algo, trainset)
    }

    with open("models/demo_svd.pkl", "wb") as f:
        pickle.dump(artifact, f)

def datasets(full=True):
    if full:
        df = pd.read_csv("data/ratings_normal_users.csv")
    else:
        df = pd.read_csv("data/ratings_normal_users_small.csv")

    rating = "rating"
    #rating = "cloglog_rating"
    #df = transform(df)
    #print(df.head(10))

    reader = Reader(rating_scale=(df[rating].min(), df[rating].max()))
    data = Dataset.load_from_df(
        df[["userId", "movieId", rating]],
        reader
    )
    trainset, testset = train_test_split(data, test_size=0.3, random_state=100)
    return (trainset, testset, data)

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
    b = algo.bi
    factors = []
    for inner_uid in range(trainset.n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        vec = p[inner_uid]
        factors.append([raw_uid, bias] + list(vec))

    num_p_cols = len(factors[0] - 2)
    p_cols = ["p" + str(i) for i in range(num_p_cols)]
    return pd.DataFrame(factors, columns=["userId", "bias"] + p_cols)

def eval_algos(data):
    params = {
        "n_factors": [4, 5, 6],
        "reg_bu": np.linspace(0.02, 0.2, 5),
        "reg_bi": np.linspace(0.02, 0.2, 5),
        "reg_pu": np.linspace(0.02, 0.2, 5),
        "reg_qi": np.linspace(0.02, 0.2, 5),
        "lr_bu": [0.002, 0.005, 0.01],
        "lr_bi": [0.002, 0.005, 0.01],
        "lr_pu": [0.002, 0.005, 0.01],
        "lr_qi": [0.002, 0.005, 0.01],
        "init_mean": np.linspace(-1, 1, 5),
        "init_std_dev": np.linspace(0.1, 1, 5),
        "n_epochs": [30],
    }
    rs = RandomizedSearchCV(
        SVDpp,
        params,
        measures=["MAE"],
        n_iter=20,
        cv=3,
        random_state=100,
        n_jobs=4,
        joblib_verbose=4,
        return_train_measures=True
    )
    rs.fit(data)

    with open("random_search_small_dataset_pp.pkl", "wb") as f:
        pickle.dump(rs, f)

    # r(u,i) = B_g + B(u) + B(i) + q(i)Tp(u)
    #        =               ... + q1(i)p1(u) + ... qn(i)pn(u)

    # Now build 

def write_grid_search():
    trainset, testset, full_dataset = datasets(full=False)
    eval_algos(full_dataset)

def read_grid_search():
    with open("random_search_small_dataset_pp.pkl", "rb") as f:
        obj = pickle.load(f)
    print(obj)
    print(obj.cv_results)

def make_hybrid_model():
    


    trainset, testset, full_dataset = datasets(full=False)
    algo = BaselineOnly()
    algo.fit(trainset)

    global_bias = trainset.global_mean
    idf = item_factors_df(algo, trainset)
    udf = user_factors_df(algo, trainset)

    #print(algo.predict(full_dataset))
    # TODO: Bring back trainset to a dataframe and find residuals



def main():
    #trainset, testset, full_dataset = datasets(full=False)
    #make_demo_algo(trainset)
    #read_grid_search()
    make_hybrid_model()



if __name__ == "__main__":
    main()