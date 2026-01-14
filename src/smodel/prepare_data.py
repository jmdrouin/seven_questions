import os
import pandas as pd

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
    return _read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)

def top_ratings():
    def make_top_ratings_dataframe():
        top_movies_ = top_movies()

        ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
        top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies_["movieId"])]

        return top_ratings
    
    return _read_or_create("data/custom/top_ratings.csv", make_top_ratings_dataframe)

def make_top_movies_dataframe():
    df = pd.read_csv('data/custom/imdb.csv')
    print(df.head())

    # Drop values that seem to have been read wrong (TODO: check why these are wrong)
    prev_length = len(df)
    df = df[df['Title'] != "#DUPE#"]
    print(prev_length - len(df), "#DUPE# entries have been dropped.")

    df.info()

    # Keep only values that have full info
    # TODO: review this. Some have just some metascore or similar missing.
    prev_length = len(df)
    df = df.dropna(subset=["Rated", "Released", "Runtime", "Writer", "Director", "Actors", "Language", "Country", "imdbRating", "imdbVotes"])
    print(prev_length - len(df), "entries with missing values have been dropped.")

    # Keep the top N
    df = df.head(N_MOVIES)

    # Merge with the base database
    movies_df = pd.read_csv('data/custom/extended_movies.csv')
    combined_df = df.merge(movies_df, left_on="imdbID", right_on="imdb_tag", how="inner")

    combined_df.info()
    print(combined_df.head())
    return combined_df

if __name__ == "__main__":
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
    for k in [0,1,5,10,20]:
        print(
            len(users[ users["divergent_ratings"] > k ]),
            "users have at least",
            k,
            "ratings above and below their average"
        )
    
    min_divergence = 15
    from sklearn.model_selection import train_test_split        
    normal_users, cold_test_users = train_test_split(
        users,
        test_size=0.1,
        random_state=100,
        shuffle=True
    )

    # Prepare a smaller set of data for quicker tests
    _, small_users = train_test_split(
        users,
        test_size=0.1,
        random_state=100,
        shuffle=True
    )

    cold_test_ratings = df[["userId","movieId","rating","timestamp"]][df["userId"].isin(cold_test_users.index)]
    normal_ratings = df[["userId","movieId","rating","timestamp"]][df["userId"].isin(normal_users.index)]
    small_ratings = df[["userId","movieId","rating","timestamp"]][df["userId"].isin(small_users.index)]

    print("Writing data/ratings_cold_users.csv... size=", len(cold_test_ratings))
    cold_test_ratings.to_csv("data/ratings_cold_users.csv", index=False)

    print("Writing data/ratings_normal_users_small.csv... size=", len(small_ratings))
    small_ratings.to_csv("data/ratings_normal_users_small.csv", index=False)

    print("Writing data/ratings_normal_users.csv... size=", len(normal_ratings))
    normal_ratings.to_csv("data/ratings_normal_users.csv", index=False)