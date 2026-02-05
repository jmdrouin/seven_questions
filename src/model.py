import pandas as pd
import numpy as np
from src import movie_info as mi

def movies_df(idf, nrows=None):
    """
    Return a dataframe of movies combining the imdb data, movieIds and preprocessed values for
    a few key columns.
    """
    movies_info_df = pd.read_csv("data/processed/top_movies.csv")
    movies_info_extended = mi.movie_info()[['Runtime_std', 'tomatoScore_std', 'age_years_std', 'age_years']]
    result = idf \
        .merge(movies_info_df, on="movieId") \
        .merge(movies_info_extended, on="movieId") \
        .sort_values(by="ratings_count", ascending=False)
    if nrows==None:
        return result
    else:
        return result.head(nrows)

def recommend_items(p: list[float], bias_weight: float, df: pd.DataFrame):
    """
    Return a dataframe of items sorted by priority of recommendation,
    in a way that fits the user taste profile "p".
    bias_weight controls the impact that the movie bias has on the recommendation.
    """
    dims = ["q"+str(i) for i in range(len(p))]
    df["pq_score"] = df[dims].to_numpy() @ np.array(p)
    df["rec_score"] = df["pq_score"] + bias_weight * df["bias"]
    sorted = df.sort_values(by="rec_score", ascending=False)
    return sorted[dims + ['Title', 'pq_score', 'rec_score', 'bias']]