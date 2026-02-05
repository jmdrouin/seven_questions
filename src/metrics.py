from sklearn.metrics import ndcg_score, root_mean_squared_error
from collections import defaultdict
import numpy as np

def ndcg(preds, k=10):
    """
    Return ndcg@k metric for preds (surprise style predictions)
    """
    by_user = defaultdict(list)
    for p in preds:
        by_user[p.uid].append(p)

    scores = []
    for uid, ps in by_user.items():
        true_values = np.array([p.r_ui for p in ps], dtype=float).reshape(1, -1)
        pred_values  = np.array([p.est  for p in ps], dtype=float).reshape(1, -1)
        score = ndcg_score(
            true_values,
            pred_values,
            k = min(k, true_values.shape[1])
        )
        scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")

def rmse(preds):
    """
    Return rmse metric for preds (surprise style predictions)
    """
    y_true = [p.r_ui for p in preds]
    y_pred = [p.est for p in preds]
    return root_mean_squared_error(y_true, y_pred)