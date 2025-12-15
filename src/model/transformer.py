# Approximate user_bias, item_bias and global_bias from the given ratings dataframe.
# Model: rating(u, i) = global_bias + user_bias(u) + item_bias(i) + residual(u, i)
# Output: (residuals, user_bias, item_bias, global_bias)

from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin

class BiasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=3):
        self._n_iter = n_iter

    def fit(self, X, y=None):
        result = _get_residuals_and_bias(X, n_iter=self._n_iter)
        (self.residuals, self.user_bias, self.item_bias) = result
        return self

    def transform(self, X):
        # TODO: This will only return something for the original data!
        cols = ["userId", "movieId", "residual", "user_bias", "global_bias", "item_bias"]
        return X.merge(self.residuals[cols], on=["userId", "movieId"], how="left")

def _get_residuals_and_bias(df, n_iter=5):
    df=df.copy()
    df['global_bias'] = 0
    df['user_bias'] = 0
    df['item_bias'] = 0

    estimation = df['global_bias'] + df['user_bias'] + df['item_bias']
    print("MSE without bias cleanup ", mean_squared_error(estimation, df['rating']))

    for i in range(n_iter):
        df = _update_bias(df)
        estimation = df['global_bias'] + df['user_bias'] + df['item_bias']
        print("MSE after iteration", i, " >> ", mean_squared_error(estimation, df['rating']))

    df['residual'] = df['rating'] - df['global_bias'] - df['user_bias'] - df['item_bias']

    # Don't return those. They can be retrieved.
    user_bias = df.groupby("userId")["user_bias"].mean().reset_index()
    item_bias = df.groupby("movieId")["item_bias"].mean().reset_index()
    return (df, user_bias, item_bias)

def _update_bias(df):
    # global bias
    df['global_bias'] = df['rating'].mean() - df['user_bias'].mean() - df['item_bias'].mean()

    # user bias:
    df = df.drop('user_bias', axis=1)
    users_df = df.groupby("userId").mean().reset_index()
    users_df['user_bias'] = users_df["rating"] - users_df["global_bias"] - users_df["item_bias"]
    df = df.merge(users_df[["userId", "user_bias"]], on="userId", how="left")

    # item bias:
    df = df.drop('item_bias', axis=1)
    item_df = df.groupby("movieId").mean().reset_index()
    item_df['item_bias'] = item_df["rating"] - item_df["global_bias"] - item_df["user_bias"]
    df = df.merge(item_df[["movieId", "item_bias"]], on="movieId", how="left")

    return df