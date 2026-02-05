#------------------------------------------------------------
# prepare_data.py
# Prepare the large csv files necessary to train a model.
#------------------------------------------------------------

import os
import pandas as pd

# Number of movies kept in the dataset
N_MOVIES = 10000

def _read_or_create(csv_file, make_dataframe):
    """Load dataframe from csv_file if it exists. Otherwise, create it by calling make_dataframe()."""
    if os.path.exists(csv_file):
        print("Reading", csv_file)
        return pd.read_csv(csv_file)
    print("Creating database", csv_file)
    df = make_dataframe()
    df.to_csv(csv_file, index=False)
    print("Database created:", csv_file)
    return df

def top_movies():
    return _read_or_create("data/processed/top_movies.csv", make_top_movies_dataframe)

def top_ratings():
    def make_top_ratings_dataframe():
        top_movies_ = top_movies()

        ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
        top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies_["movieId"])]

        return top_ratings
    return _read_or_create("data/processed/top_ratings.csv", make_top_ratings_dataframe)

def make_top_movies_dataframe():
    df = pd.read_csv('data/processed/imdb.csv')
    print(df.head())

    # Drop values that seem to have been read wrong when using IMDB's api
    prev_length = len(df)
    df = df[df['Title'] != "#DUPE#"]
    print(prev_length - len(df), "#DUPE# entries have been dropped.")

    df.info()

    # Keep only values that have full info
    prev_length = len(df)
    relevant_fields = [
        "Rated", "Released", "Runtime", "Writer", "Director", "Actors",
        "Language", "Country", "imdbRating", "imdbVotes"
    ]
    df = df.dropna(subset=relevant_fields)
    print(prev_length - len(df), "entries with missing values have been dropped.")

    # Keep the top N
    df = df.head(N_MOVIES)

    # Merge with the base database
    movies_df = pd.read_csv('data/processed/extended_movies.csv')
    combined_df = df.merge(movies_df, left_on="imdbID", right_on="imdb_tag", how="inner")

    combined_df.info()
    print(combined_df.head())
    return combined_df

def main():
    # Select users that have at least some good and bad ratings,
    # Save those ratings for building the model
    
    df = top_ratings()
    users = df \
        .groupby("userId")["rating"] \
        .agg(avg_rating="mean", num_ratings="count")
    
    print("Number of users:", len(users))
    
    df = df.merge(users, on="userId")
    df["is_high"] = df["rating"] > df["avg_rating"]
    df["is_low"] = df["rating"] < df["avg_rating"]

    user_relative_count = df \
        .groupby("userId") \
        .agg(
            below_avg_count=("is_high", "sum"),
            above_avg_count=("is_low", "sum"),
        )
    
    users = users \
        .merge(user_relative_count, on="userId")
    
    users["divergent_ratings"] = users[["below_avg_count", "above_avg_count"]].min(axis=1)

    n_users = len(users)
    for k in [0,1,5,10,20,25]:
        print(
            len(users[ users["divergent_ratings"] > k ]),
            "users have at least",
            k,
            "ratings above and below their average"
        )
    
    users_100k = users \
        .sort_values("divergent_ratings", ascending=False) \
        .head(100000)
    
    ratings_100k_users = df \
        [ df["userId"].isin(users_100k.index) ] \
        .rename({"movieId": "itemId"}, axis=1) \
        [ ["userId", "itemId", "rating"] ]
    
    print("Number of ratings:", len(ratings_100k_users))
    print("Number of users:", len(users_100k))
    print(users_100k.tail())
    # Each user has at least 28 divergent ratings
    # and an average of 267 ratings.
    # Taking 27 ratings per user for testing
    # means around 10% test data.

    # Smaller set for faster computation (using mid-sized users)
    users_20k = users \
        .sort_values("divergent_ratings", ascending=False) \
        .head(50000) \
        .tail(20000)
    ratings_20k_users = df \
        [ df["userId"].isin(users_20k.index) ] \
        .rename({"movieId": "itemId"}, axis=1) \
        [ ["userId", "itemId", "rating"] ]

    ratings_100k_test = ratings_100k_users \
        .groupby("userId", group_keys=False) \
        .sample(n=27, random_state=100)
    ratings_100k_train = ratings_100k_users \
        .drop(ratings_100k_test.index)

    ratings_20k_test = ratings_20k_users \
        .groupby("userId", group_keys=False) \
        .sample(n=27, random_state=100)
    ratings_20k_train = ratings_20k_users \
        .drop(ratings_20k_test.index)
    
    def write_csv(file, data):
        dest = "data/processed/" + file
        print("Writing to", file, "-- size =", len(data))
        data.to_csv(dest + "ratings_100k_users_train.csv", index=False)
    
    write_csv("ratings_100k_users_train.csv", ratings_100k_train)
    write_csv("ratings_100k_users_test.csv", ratings_100k_test)
    write_csv("ratings_20k_users_train.csv", ratings_20k_train)
    write_csv("ratings_20k_users_test.csv", ratings_20k_test)

if __name__ == "__main__":
    main()