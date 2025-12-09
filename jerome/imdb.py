# Database imdb.csv has several issues in its raw form. This script deletes broken entries
# and those missing important data, and merges it with our non-imdb data.

# Keep only the top movies:
N_MOVIES = 10000

import pandas as pd

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