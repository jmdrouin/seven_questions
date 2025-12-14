import os
import pandas as pd

def _read_or_create(csv_file, make_dataframe):
    """Load dataframe from csv_file if it exists. Otherwise, create it by calling make_dataframe()."""
    if os.path.exists(csv_file):
        print("Reading", csv_file)
        return pd.read_csv(csv_file)
    print("Creating database", csv_file)
    df = make_dataframe()
    df.to_csv(csv_file)
    print("Database created:", csv_file)
    return df

def top_ratings():
    def make_top_ratings_dataframe():
        # Dataframe restricted to the top 10K users and top 10K movies
        top_movies = _read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)

        ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
        top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies["movieId"])]
        print("All ratings:", len(ratings_df), "-- Ratings of top movies:", len(top_ratings))

        counts = top_ratings['userId'].value_counts()
        top_users = counts.head(1000).index
        top_ratings = top_ratings[top_ratings['userId'].isin(top_users)]

        return top_ratings
    
    return _read_or_create("data/custom/top_ratings.csv", make_top_ratings_dataframe)


def make_top_ratings_dataframe(n_users=1000):
    # Dataframe restricted to the top n_users users and top 10K movies
    top_movies = _read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)

    ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
    top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies["movieId"])]
    print("All ratings:", len(ratings_df), "-- Ratings of top movies:", len(top_ratings))

    counts = top_ratings['userId'].value_counts()
    top_users = counts.head(n_users).index
    top_ratings = top_ratings[top_ratings['userId'].isin(top_users)]

    return top_ratings
