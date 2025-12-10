from util import read_or_create
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from imdb import make_top_movies_dataframe
import matplotlib.pyplot as plt
from bias import get_residuals_and_bias
import numpy as np


def main():
    df = top_ratings()

    (residuals, user_bias, item_bias) = get_residuals_and_bias(df)
    num_components = 4
    movies_df = find_dimensions(residuals, user_bias, item_bias, num_components)
    explore_results(movies_df)

def explore_results(movies_df):
    dim_cols = [c for c in movies_df.columns if c[:4] == "dim_"]
    full_movies_df = read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)
    full_movies_df = full_movies_df.merge(movies_df, on="movieId")

    #find_points(full_movies_df, dim_cols)
    find_clusters(full_movies_df, dim_cols)
    return

    print(full_movies_df[dim_cols].describe())

    good_movies = full_movies_df.sort_values(by="ratings_count").tail(1000)
    cols = ['Title', 'item_bias', 'Genre', 'ratings_count'] + dim_cols
    good_movies = good_movies[cols]

    good_movies["excentricity"] = (good_movies[dim_cols]**2).sum(axis=1)

    print("\n\n ---- ISO 0+ ----")
    good_movies["tmp"] = good_movies["excentricity"] - good_movies["dim_0"]**3
    print(good_movies.sort_values(by="tmp").head())

    print("\n\n ---- ISO 0- ----")
    good_movies["tmp"] = good_movies["excentricity"] + good_movies["dim_0"]**3
    print(good_movies.sort_values(by="tmp").head())

    print("\n\n ---- AVERAGE MOVIES ----")
    print(good_movies.sort_values(by="excentricity").head())

    for col in dim_cols:
        sorted = good_movies.sort_values(by=col)
        print("\n -----", col, "---------------------")
        print("BOTTOM:", list(sorted.head()['Title']))
        print("TOP:", list(sorted.tail()['Title']))

    #show_scatter_plot(good_movies, x = 'dim_0', y = 'dim_2', n = 100)

def find_clusters(movies_df, dim_cols):
    from sklearn.cluster import KMeans
    k = 10   # choose number of clusters
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(movies_df[dim_cols])
    movies_df["cluster"] = labels
    print(movies_df.head())

    import matplotlib.pyplot as plt

    df = movies_df.sort_values(by="ratings_count").tail(100)
    plt.figure(figsize=(6,6))
    plt.scatter(df['dim_0'], df['dim_1'], c=df['cluster'], cmap='tab10', s=40)
    plt.title("Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    for _, r in df.iterrows():
        plt.annotate(r['Title'], (r['dim_0'], r['dim_1']), textcoords="offset points", xytext=(5,5))
    plt.show()


def find_points(movies_df, dim_cols):
    df = movies_df[['movieId', 'Title', 'ratings_count'] + dim_cols]
    df = df.sort_values(by="ratings_count").tail(200).reset_index()
    print("find points...")
    print(df)

    main_movies = df.tail(1)
    print(main_movies)
    position = main_movies[dim_cols].values[0]

    df['dist'] = df[dim_cols].sub(position).pow(3).sum(axis=1)
    df['dist_min'] = df['dist']

    for i in range(9):
        is_main = df['movieId'].isin(main_movies['movieId'])
        idx = df.loc[~is_main, 'dist_min'].idxmax()
        main_movies = pd.concat([main_movies, df.loc[[idx]]], ignore_index=True)
        position = df.loc[idx, dim_cols].values
        df['dist'] = df[dim_cols].sub(position).pow(2).sum(axis=1)
        df['dist_min'] = df[['dist', 'dist_min']].min(axis=1)

    print(main_movies)

def show_scatter_plot(movies_df, x = 'dim_0', y = 'dim_2', n = 100):
    top_n = movies_df.sort_values(by="ratings_count").tail(n)
    plt.figure(figsize=(6,6))
    plt.scatter(top_n[x], top_n[y])
    for _, r in top_n.iterrows():
        plt.annotate(r['Title'], (r[x], r[y]), textcoords="offset points", xytext=(5,5))
    plt.show()

def top_ratings():
    def make_top_ratings_dataframe():
        # Dataframe restricted to the top 10K users and top 10K movies
        top_movies = read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)

        ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
        top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies["movieId"])]
        print("All ratings:", len(ratings_df), "-- Ratings of top movies:", len(top_ratings))

        counts = top_ratings['userId'].value_counts()
        top_users = counts.head(1000).index
        top_ratings = top_ratings[top_ratings['userId'].isin(top_users)]

        return top_ratings
    
    return read_or_create("data/custom/top_ratings.csv", make_top_ratings_dataframe)

# Run a truncatedSVD on ratings to identify main dimensions of movies.
# return a movies dataframe merged with those dimensions as columns: dim_0, dim_1...
def find_dimensions(ratings_df, users_df, movies_df, num_components):
    (sparse_ratings, users_df, movies_df) = get_sparse_ratings(ratings_df, users_df, movies_df)
    movies_df = reduce_dimensions(sparse_ratings, movies_df, num_components)
    movies_df = movies_df.drop(["movie_code"], axis=1)
    return movies_df

def reduce_dimensions(sparse_ratings, item_bias, num_components):
    k=num_components
    svd = TruncatedSVD(n_components=k, random_state=0)
    movie_embeddings = svd.fit_transform(sparse_ratings)
    movie_embeddings_df = pd.DataFrame(
        movie_embeddings,
        columns=["dim_" + str(i) for i in range(k)]
    )
    movie_embeddings_df.index.name = "movie_code"
    item_bias = item_bias.merge(movie_embeddings_df, on="movie_code")
    return item_bias

# Return a sparse matrix (coo_matrix) from a ratings dataframe.
# To avoid empty columns and rows, we create user_code and movie_code,
# which act like ids but have no gaps (ids 0,1,3,5 become codes 0,1,2,3)
# Output: (sparse_ratings, users_df, movies_df)
# where users_df has received a user_code column,
# and movies_df has received a movie_code column.
def get_sparse_ratings(ratings_df, users_df, movies_df):
    users_df['user_code'] = users_df["userId"].astype("category").cat.codes
    movies_df['movie_code'] = movies_df["movieId"].astype("category").cat.codes
    ratings_df = ratings_df.merge(users_df, on="userId", how="left")
    ratings_df = ratings_df.merge(movies_df, on="movieId", how="left")

    print(ratings_df.head())

    sparse_ratings = coo_matrix(
        (ratings_df["residual"].astype(float), (ratings_df["movie_code"], ratings_df["user_code"])),
        shape=(len(movies_df), len(users_df))
    )

    return (sparse_ratings, users_df, movies_df)

if __name__ == "__main__":
    main()