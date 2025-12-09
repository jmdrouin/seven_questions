from util import read_or_create
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from imdb import make_top_movies_dataframe
import matplotlib.pyplot as plt

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

def main():
    df = read_or_create("data/custom/top_ratings.csv", make_top_ratings_dataframe)
    
    (residuals, user_bias, item_bias, global_bias) = get_residuals_and_bias(df)
    (sparse_ratings, user_bias, item_bias) = get_sparse_ratings(residuals, user_bias, item_bias)
    k = 8
    movies_df = reduce_dimensions(sparse_ratings, item_bias, k)
    print(movies_df.head())

    dim_cols = ["dim_" + str(i) for i in range(k)]
    scaler = StandardScaler()
    movies_df[dim_cols] = scaler.fit_transform(movies_df[dim_cols])
    print(movies_df.head())

    full_movies_df = read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)
    full_movies_df = full_movies_df.merge(movies_df, on="movieId")
    print(full_movies_df.head())

    print(full_movies_df.describe())

    good_movies = full_movies_df.sort_values(by="ratings_count").tail(1000)
    cols = ['Title', 'item_bias', 'Genre', 'ratings_count'] + dim_cols
    good_movies = good_movies[cols]

    good_movies["excentricity"] = (good_movies[dim_cols]**2).sum(axis=1)

    print("\n\n ---- MOST_RATED ----")
    print(good_movies.sort_values(by="ratings_count").tail(50))

    print("\n\n ---- ISO 0+ ----")
    good_movies["tmp"] = good_movies["excentricity"] - good_movies["dim_0"]**3
    print(good_movies.sort_values(by="tmp").head())

    print("\n\n ---- ISO 0- ----")
    good_movies["tmp"] = good_movies["excentricity"] + good_movies["dim_0"]**3
    print(good_movies.sort_values(by="tmp").head())

    print("\n\n ---- AVERAGE MOVIES ----")
    print(good_movies.sort_values(by="excentricity").head())


    for dim in range(k):
        col = "dim_"+str(dim)
        sorted = good_movies.sort_values(by=col)
        sorted['m'] = abs(sorted[col])

        print("\n\n ----- DIMENSION", dim, "---------------------")
        print("BOTTOM:", list(sorted.head()['Title']))
        #print("MIDDLE:", list(sorted.sort_values(by='m').head()['Title']))
        print("TOP:", list(sorted.tail()['Title']))


    top_n = good_movies.sort_values(by="ratings_count").tail(100)
    plt.figure(figsize=(6,6))
    x = 'dim_0'
    y = 'dim_2'
    z = 'dim_2'
    plt.scatter(top_n[x], top_n[y])#, c=top_n[z], cmap="viridis")
    for _, r in top_n.iterrows():
        plt.annotate(r['Title'], (r[x], r[y]), textcoords="offset points", xytext=(5,5))
    plt.show()

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
    
def get_residuals_and_bias(df):
    global_bias = df['rating'].mean()
    df['rating_0'] = df['rating'] - global_bias
    
    # USER_BIAS:
    user_bias = df.groupby("userId")["rating_0"].mean().reset_index()
    user_bias = user_bias.rename({"rating_0": "user_bias"}, axis=1)
    df = df.merge(user_bias, on="userId", how="left")
    df["rating_1"] = df["rating_0"] - df["user_bias"]

    # FILM_BIAS
    item_bias = df.groupby("movieId")["rating_1"].mean().reset_index()
    item_bias = item_bias.rename({"rating_1": "item_bias"}, axis=1)
    df = df.merge(item_bias, on="movieId", how="left")
    df["rating_2"] = df["rating_1"] - df["user_bias"]

    # ESTIMATION OF QUALITY
    df["estimation"] = global_bias + df["user_bias"] + df["item_bias"]
    df["residual"] = df["rating"] - df["estimation"]

    residuals = df[["userId", "movieId", "residual"]]
    return (residuals, user_bias, item_bias, global_bias)

def get_sparse_ratings(ratings_df, users_df, movies_df):
    # Create sparse matrix
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