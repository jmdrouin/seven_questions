import pandas as pd

ratings = pd.read_csv("data/ml-32m/ratings.csv")

# Ratings distribution
rating_dist = (
    ratings.groupby("rating")
    .size()
    .reset_index(name="count")
)
rating_dist.to_csv("rating_distribution.csv", index=False)

# Ratings per user
ratings_per_user = (
    ratings.groupby("userId")
    .size()
    .reset_index(name="n_ratings")
)
ratings_per_user.to_csv("ratings_per_user.csv", index=False)

# Ratings per movie
ratings_per_movie = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="n_ratings")
)
ratings_per_movie.to_csv("ratings_per_movie.csv", index=False)

# Top movies and users
N_USERS = 1000
N_MOVIES = 1000
top_users = ratings["userId"].value_counts().head(N_USERS).index
top_movies = ratings["movieId"].value_counts().head(N_MOVIES).index
filtered = ratings[
    ratings["userId"].isin(top_users) &
    ratings["movieId"].isin(top_movies)
].copy()
filtered.to_csv("top_1000ui_ratings.csv", index=False)

# Average rating per user
user_means = (
    ratings.groupby("userId")["rating"]
    .mean()
    .reset_index(name="avg_rating")
)
user_means.to_csv("user_avg_rating.csv", index=False)

# Average rating per movie
movie_means = (
    ratings.groupby("movieId")["rating"]
    .mean()
    .reset_index(name="avg_rating")
)
movie_means.to_csv("movie_avg_rating.csv", index=False)

# above below
user_mean = ratings.groupby("userId")["rating"].mean().rename("user_mean")
df = ratings.join(user_mean, on="userId")
df["above"] = df["rating"] > df["user_mean"]
df["below"] = df["rating"] < df["user_mean"]
user_signal = (
    df.groupby("userId")
    .agg(
        n_above=("above", "sum"),
        n_below=("below", "sum"),
        n_total=("rating", "size"),
    )
    .reset_index()
)
user_signal["min_above_below"] = user_signal[["n_above", "n_below"]].min(axis=1)
user_signal[["userId", "min_above_below", "n_total"]].to_csv(
    "user_signal_strength.csv",
    index=False
)