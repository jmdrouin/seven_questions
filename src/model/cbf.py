import pandas as pd
import numpy as np
from dataframes import top_movies
from sklearn.metrics.pairwise import cosine_similarity

def section(name):
    print("\n====================================================")
    print("=   ", name)
    print("====================================================")

def main():
    rdf = pd.read_csv("data/custom/top_ratings.csv")
    mdf = movies_df()

    userId = rdf.head(1)['userId'].values[0]

    mdf = mdf.dropna()
    rec = recommend(userId, rdf, mdf.copy(), K=10)

    mdf["score"] = rec
    mdf = mdf.merge(top_movies(), on="movieId")[["Title", "movieId", "score"]]

    section("Recommended films (unfiltered) for userid " + str(userId))
    print(mdf.sort_values(by="score").tail(10))

    return

def build_user_profile(user_id, rdf, mdf, mu=3.0):
    rdf = rdf[rdf["userId"] == user_id]
    rdf['weight'] = rdf['rating'] - mu
    rdf = rdf[rdf['weight'] >= 1] # only keep significant weights (positive reviews)
    rdf = rdf.dropna()
    if len(rdf) == 0: return None

    df = rdf.merge(mdf, on="movieId")
    df = df.drop(["userId", "movieId", "rating", "timestamp"], axis=1)
    total_weight = np.sum(np.abs(df["weight"]))
    df_w = df.mul(df["weight"], axis=0) / total_weight
    df_w = df_w.sum()
    df_w = df_w.drop(["weight"])
    return df_w

def recommend(user_id, rdf, mdf, K=10):
    profile = build_user_profile(user_id, rdf, mdf, mu=3.0)
    if profile is None: return []
    profile = profile.reindex(mdf.columns).to_numpy().reshape(1, -1)
    scores = cosine_similarity(mdf, profile).ravel()
    return scores

def movies_df():
    relevant_cols = ['movieId', 'Title', 'Rated', 'Released', 'Runtime',
       'Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country',
       'Awards', 'BoxOffice', 'Metascore', 'imdbRating', 'imdbVotes',
       'tomatoScore', 'genres']
    df = top_movies()[relevant_cols].set_index("movieId")
    df = df.drop(["Title", "Director", "Writer", "Actors", "BoxOffice", "imdbVotes"], axis=1)

    df = encode_ratings(df, verbose=False)
    df = encode_age(df)

    df = encode_list(df, "Genre", "imdb_genre_", min_size=50)
    df = encode_list(df, "genres", "ml_genre_", min_size=50)
    df = encode_list(df, "Country", "country_", min_size=200)

    df = encode_first(df, "Language", "ml")

    df = encode_awards(df)

    #print(df.info())

    #corr = df.corr()
    #mask = np.triu(np.ones_like(corr, dtype=bool))
    #corr_lower = corr.mask(mask)
    #top_corr = (
    #    corr_lower.unstack()\
    #        .sort_values(ascending=False)
    #)

    #import matplotlib.pyplot as plt
    #plt.hist(df['Runtime'], bins=30)
    #plt.show()

    df = transform(df)
    return df

def transform(df):
    from sklearn.preprocessing import StandardScaler
    rscaler = StandardScaler()
    rcols = ['Runtime', 'Metascore', 'imdbRating', 'tomatoScore', "age_years"]
    rscaler.fit(df[rcols])
    df[rcols] = rscaler.transform(df[rcols])

    from sklearn.preprocessing import FunctionTransformer
    log_scaler = FunctionTransformer(np.log1p, validate=False)
    lcols = ["Oscars", "Awards_won", "Nominations"]
    df[lcols] = log_scaler.transform(df[lcols])

    return df

def encode_awards(original_df):
    df = original_df.copy()[["Awards"]]

    df['Oscars'] = df['Awards']\
        .str.extract(r'Won\s+(\d+)\s+Oscar', expand=False)\
        .fillna(0)\
        .astype(int)
    
    df['Awards_won'] = df['Awards']\
        .str.extract(r'(\d+)\s+win', expand=False)\
        .fillna(0)\
        .astype(int)
    
    df['Nominations'] = df['Awards']\
        .str.extract(r'(\d+)\s+nomination', expand=False)\
        .fillna(0)\
        .astype(int)
    
    original_df = original_df.drop(["Awards"], axis=1)
    original_df = original_df.merge(df, on="movieId")
    original_df = original_df.drop(["Awards"], axis=1)
    return original_df

def encode_first(df, col, prefix):
    df[col] = df[col].str.extract(r'^([^|]+)')
    return encode_list(df, col, prefix, min_size=50)

def encode_list(df, col, prefix, min_size=100):
    encoded = df[col].str.get_dummies(sep="|").add_prefix(prefix)
    df = df.drop([col], axis=1)
    encoded = encoded.loc[:, encoded.sum(axis=0) >= min_size]

    return df.merge(encoded, on="movieId")

def encode_age(df):
    df["Released"] = pd.to_datetime(
        df["Released"],
        format="%d %b %Y",
        errors="coerce"
    )
    ref_date = pd.Timestamp("2026-01-01")
    df["age_years"] = (ref_date - df["Released"]).dt.days / 365.25
    return df.drop(["Released"], axis=1)

def encode_ratings(df, verbose=True):
    if verbose:
        section("MPAA RATINGS")
        print(df['Rated'].value_counts())

    rating_map = {
        # 1 — Kids
        "G": "G",
        "TV-Y": "G",
        "TV-Y7-FV": "G",
        "TV-G": "G",

        # 2 — Family / PG
        "PG": "PG",
        "PG-13": "PG",
        "TV-PG": "PG",
        "TV-13": "PG",
        "GP": "PG",
        "M": "PG",
        "M/PG": "PG",
        "TV-14": "PG", # This is teenage, but we have only a few

        # 4 — Adults
        "R": "R",
        "TV-MA": "R",
        "NC-17": "R",
        "X": "R",
        "16+": "R",
        "18+": "R",

        # 5 — Unrated / Legacy
        "Not Rated": "unrated",
        "Unrated": "unrated",
        "Approved": "unrated",
        "Passed": "unrated"
    }
    df['Rated'] = df['Rated'].map(rating_map)
    if verbose:
        print("\n--After simplification:")
        print(df['Rated'].value_counts())

    dummies = pd.get_dummies(
        df["Rated"],
        prefix="rating",
        dtype=int
    )
    df = df.drop(["Rated"], axis=1)
    return df.merge(dummies, on="movieId")

if __name__ == "__main__":
    main()