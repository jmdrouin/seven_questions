import os
import pandas as pd

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

def top_ratings():
    def make_top_ratings_dataframe():
        # Dataframe restricted to the top 10K users and top 10K movies
        top_movies = top_movies()

        ratings_df = pd.read_csv('data/ml-32m/ratings.csv')
        top_ratings = ratings_df[ratings_df["movieId"].isin(top_movies["movieId"])]
        print("All ratings:", len(ratings_df), "-- Ratings of top movies:", len(top_ratings))

        counts = top_ratings['userId'].value_counts()
        top_users = counts.head(1000).index
        top_ratings = top_ratings[top_ratings['userId'].isin(top_users)]

        return top_ratings
    
    return _read_or_create("data/custom/top_ratings.csv", make_top_ratings_dataframe)

def top_movies():
    return _read_or_create("shared_data/top_movies.csv", make_top_movies_dataframe)

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