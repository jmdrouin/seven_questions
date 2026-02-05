#------------------------------------------------------------
# process_app_data.py
# Preprocess data needed for the streamlit app.
#------------------------------------------------------------

import pandas as pd

# Destination folder
dest = "app/preprocessed_data/"

# Data source (large file)
ratings = pd.read_csv("data/ml-32m/ratings.csv")

# Lighter version (first 1000 rows)
pd.read_csv("data/ml-32m/ratings.csv", nrows=1000) \
    .to_csv(dest + "ratings1000.csv")

# Ratings distribution
ratings.groupby("rating") \
    .size() \
    .reset_index(name="count") \
    .to_csv(dest + "rating_distribution.csv", index=False)

# Ratings per user
ratings.groupby("userId") \
    .size() \
    .reset_index(name="n_ratings") \
    .to_csv(dest + "ratings_per_user.csv", index=False)

# Ratings per movie
ratings.groupby("movieId") \
    .size() \
    .reset_index(name="n_ratings") \
    .to_csv(dest + "ratings_per_movie.csv", index=False)

# Top movies and users
N_USERS = 1000
N_MOVIES = 1000
top_users = ratings["userId"].value_counts().head(N_USERS).index
top_movies = ratings["movieId"].value_counts().head(N_MOVIES).index
filtered = ratings[
    ratings["userId"].isin(top_users) &
    ratings["movieId"].isin(top_movies)
].copy()
filtered.to_csv(dest + "top_1000ui_ratings.csv", index=False)

# Average rating per user
ratings.groupby("userId") \
    ["rating"] \
    .mean() \
    .reset_index(name="avg_rating") \
    .to_csv(dest + "user_avg_rating.csv", index=False)

# Average rating per movie
ratings.groupby("movieId") \
    ["rating"] \
    .mean() \
    .reset_index(name="avg_rating") \
    .to_csv(dest + "movie_avg_rating.csv", index=False)

# Signal Strength (above and below average)
user_mean = ratings.groupby("userId")["rating"].mean().rename("user_mean")
df = ratings.join(user_mean, on="userId")
df["above"] = df["rating"] > df["user_mean"]
df["below"] = df["rating"] < df["user_mean"]
user_signal = df \
    .groupby("userId") \
    .agg(
        n_above=("above", "sum"),
        n_below=("below", "sum"),
        n_total=("rating", "size"),
    ) \
    .reset_index()
user_signal["min_above_below"] = user_signal[["n_above", "n_below"]].min(axis=1)
user_signal[["userId", "min_above_below", "n_total"]].to_csv(
    dest + "user_signal_strength.csv",
    index=False
)