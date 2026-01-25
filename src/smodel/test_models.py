import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, NormalPredictor, BaselineOnly
from sklearn.model_selection import train_test_split

def cloglog(df, lower_bound=0, upper_bound=6):
    zero_to_one_rating = (df['rating'] - lower_bound) / (upper_bound - lower_bound)

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

    return np.log(-np.log(1 - zero_to_one_rating))

def predictor(model):
    return lambda r: model.predict(r["userId"], r["itemId"]).est

def get_split_data():
    df = pd \
        .read_csv("data/ratings_normal_users_small.csv") \
        .drop("timestamp", axis=1) \
        .rename({"movieId": "itemId"}, axis=1) \
        [["userId", "itemId", "rating"]]
    
    n = len(df)

    df_sampled = df \
        .groupby("userId", group_keys=False) \
        .sample(n=1, random_state=100)

    return df_sampled

def main():
    df = get_split_data()
    print(df.head())
    return


    df_train, df_test = train_test_split(df, test_size=0.25, random_state=100)
    reader = Reader(rating_scale=(0.5, 5))

    data_train = Dataset.load_from_df(df_train, reader)
    trainset = data_train.build_full_trainset()

    #data_test = Dataset.load_from_df(df_test, reader)

    # 1. Random values
    print("Random model...")
    random_model = NormalPredictor()
    random_model.fit(trainset)
    df_train["random_prediction"] = df_train.apply(predictor(random_model), axis=1)
    df_test["random_prediction"] = df_test.apply(predictor(random_model), axis=1)
    
    # 2. Baseline
    # TODO: Test different params
    print("Baseline model...")
    baseline_model = BaselineOnly()
    baseline_model.fit(trainset)
    df_train["baseline_prediction"] = df_train.apply(predictor(baseline_model), axis=1)
    df_test["baseline_prediction"] = df_test.apply(predictor(baseline_model), axis=1)

    # 3. SVD(10)





if __name__ == "__main__":
    main()