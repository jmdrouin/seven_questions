# Extend movies db to facilitate imdb data fetching

import pandas as pd
from scipy.stats import skew

def get_ratings_summary():
    ratings_df = pd.read_csv("data/ml-32m/ratings.csv")
    return ratings_df.groupby("movieId")["rating"].agg(
        ratings_count = "count",
        ratings_mean = "mean",
        ratings_median = "median",
        ratings_var = "var",
        ratings_skewness = skew
    ).reset_index()

def make_extended_movies_df():
    # Dataframe with basic ratings data for each movie:
    summary_df = get_ratings_summary()

    # Add movies data to it (tags, title...)
    movies_df = pd.read_csv("data/ml-32m/movies.csv")
    extended_df = movies_df.merge(summary_df, on="movieId", how="left")

    # Add ids to use with external apis
    links_df = pd.read_csv("data/ml-32m/links.csv")
    extended_df = extended_df.merge(links_df, on="movieId")

    # Fix the imdb tag to have the form "tt0123456"
    extended_df['imdb_tag'] = extended_df['imdbId'].apply(lambda x: f"tt{x:07d}")
    extended_df = extended_df.drop(["imdbId", "tmdbId"], axis=1)

    # Sort by number of ratings so it's easier to fetch the most important films first
    extended_df = extended_df.sort_values("ratings_count", ascending=False).reset_index()
    extended_df.to_csv("data/custom/extended_movies.csv", index=False)

    return extended_df

if __name__ == "__main__":
    df = make_extended_movies_df()
    df.to_csv("data/custom/extended_movied.csv")