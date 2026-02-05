import pandas as pd

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
