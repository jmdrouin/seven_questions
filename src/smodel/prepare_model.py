import pandas as pd
from surprise import SVD, SVDpp, accuracy, Dataset, Reader, NormalPredictor, BaselineOnly
#from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import pickle

def transform(df):
    # Put ratings in the 0-1 range
    lower_bound = -1
    upper_bound = 6.5
    df['unit_rating'] = (df['rating'] - lower_bound) / (upper_bound - lower_bound)

    # Suppose we have influencing factors F0, F1, ... Fn, all between -inf and inf.
    #      r = B^(e^F0)^...^(e^Fn)
    #        = B^( e^(F0+...Fn) )
    # If the base is B = (1 - 1/e), then
    # -log(1-r) = -e^(F0+...+Fn) log(1/e)
    #           = e^(F0+...+Fn)
    # log( -log(1-r) ) = (F0+...+Fn) log(e)
    #                  = F0 + ... + Fn
    # And the problem of exponential factors becomes one of sum of factors.
    # (complementary log-log function -- cloglog)
    import numpy as np
    df["cloglog_rating"] = np.log(-np.log(1 - df["unit_rating"]))

    return df

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
    return (trainset, testset)

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
    factors = []
    for inner_uid in range(trainset.n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        vec = p[inner_uid]
        factors.append([raw_uid] + list(vec))

    cols = ["p" + str(i) for i in range(len(factors[0]) - 1)]
    return pd.DataFrame(factors, columns=["userId"] + cols)

def eval_algos(trainset, testset):
    def eval_fcp(name, algo, trainset, testset):
        print(name, "...")
        algo.fit(trainset)
        preds = algo.test(testset)
        print(f"{name:12s} FCP = {accuracy.fcp(preds, verbose=False):.4f}")
        print(f"{name:12s} MAE = {accuracy.mae(preds, verbose=False):.4f}")

    eval_fcp("Random", NormalPredictor(), trainset, testset)
    eval_fcp("Baseline", BaselineOnly(bsl_options={"n_epochs":3}), trainset, testset)
    for n_factors in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
        eval_fcp("SVD-"+str(n_factors), SVD(n_factors=n_factors, n_epochs=25), trainset, testset)

    # r(u,i) = B_g + B(u) + B(i) + q(i)Tp(u)
    #        =               ... + q1(i)p1(u) + ... qn(i)pn(u)

    # Now build 

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

def main():
    trainset, testset = datasets(full=False)
    make_demo_algo(trainset)
    #eval_algos(trainset, testset)


if __name__ == "__main__":
    main()