from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

# Run a truncatedSVD on ratings to identify main dimensions of movies.
# return a movies dataframe merged with those dimensions as columns: dim_0, dim_1...
def find_dimensions(ratings_df, num_components):
    (sparse_ratings, users_df, movies_df) = _get_sparse_ratings(ratings_df)
    (movies_df, users_df) = _reduce_dimensions(sparse_ratings, movies_df, users_df, num_components)
    movies_df = movies_df.drop(["movie_code"], axis=1)
    users_df = users_df.drop(["user_code"], axis=1)
    return (movies_df, users_df)

def _reduce_dimensions(sparse_ratings, item_bias, user_bias, num_components):
    k=num_components
    svd = TruncatedSVD(n_components=k, random_state=100)

    movie_embeddings = svd.fit_transform(sparse_ratings)
    user_factors = svd.components_.T

    # Varimax rotation:
    from factor_analyzer.rotator import Rotator
    rotator = Rotator(method="varimax")
    rotated_movie_embeddings = rotator.fit_transform(movie_embeddings)
    rotated_user_factors = user_factors @ rotator.rotation_
    movie_embeddings = rotated_movie_embeddings
    user_factors = rotated_user_factors

    # Merge back user dims
    user_embeddings_df = pd.DataFrame(
        user_factors,
        columns=["user_dim_" + str(i) for i in range(k)]
    )
    user_embeddings_df["user_code"] = np.arange(len(user_factors))
    user_bias = user_bias.merge(user_embeddings_df, on="user_code")

    # Merge back movie dims
    movie_embeddings_df = pd.DataFrame(
        movie_embeddings,
        columns=["dim_" + str(i) for i in range(k)]
    )
    movie_embeddings_df["movie_code"] = np.arange(len(movie_embeddings))
    item_bias = item_bias.merge(movie_embeddings_df, on="movie_code")

    return (item_bias, user_bias)

# Return a sparse matrix (coo_matrix) from a ratings dataframe.
# To avoid empty columns and rows, we create user_code and movie_code,
# which act like ids but have no gaps (ids 0,1,3,5 become codes 0,1,2,3)
# Output: (sparse_ratings, users_df, movies_df)
# where users_df has received a user_code column,
# and movies_df has received a movie_code column.
def _get_sparse_ratings(df):
    df["user_code"] = df["userId"].astype("category").cat.codes
    users_df = df[["userId", "user_code"]].groupby("userId").first()
    users_df["userId"] = users_df.index

    df["movie_code"] = df["movieId"].astype("category").cat.codes
    movies_df = df[["movieId", "movie_code"]].groupby("movieId").first()
    movies_df["movieId"] = movies_df.index

    sparse_ratings = coo_matrix(
        (df["residual"].astype(float), (df["movie_code"], df["user_code"])),
        shape=(len(movies_df), len(users_df))
    )

    return (sparse_ratings, users_df, movies_df)