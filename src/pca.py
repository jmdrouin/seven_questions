import numpy as np
from sklearn.decomposition import PCA
from surprise.prediction_algorithms.predictions import Prediction

class PcaSvd:
    def __init__(self, base_algo, n_components, random_state=100):
        pca = PCA(n_components=n_components, random_state=random_state).fit(base_algo.qi)
        W = pca.components_.T
        self.bu = base_algo.bu
        self.bi = base_algo.bi
        self.qi = base_algo.qi @ W
        self.pu = base_algo.pu @ W

def pca_predict(pca_algo, trainset, testset):
    mu = trainset.global_mean

    preds = []
    for uid, iid, r_ui in testset:
        try:
            u = trainset.to_inner_uid(uid)
            bu = pca_algo.bu[u]
            pu = pca_algo.pu[u]
            known_u = True
        except ValueError:
            # User is absent from set
            bu = 0.0
            pu = None
            known_u = False

        try:
            i = trainset.to_inner_iid(iid)
            bi = pca_algo.bi[i]
            qi = pca_algo.qi[i]
            known_i = True
        except ValueError:
            # Item is absent from set
            bi = 0.0
            qi = None
            known_i = False

        dot = float(np.dot(pu, qi)) if (known_u and known_i) else 0.0
        est = float(mu + bu + bi + dot)

        preds.append(Prediction(uid, iid, r_ui, est, details={}))

    return preds