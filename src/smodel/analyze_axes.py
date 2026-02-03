from src.smodel import model as M
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

def find_clusters(df, dims, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(df[dims])
    centers = kmeans.cluster_centers_
    df["cluster"] = labels
    return df

def find_representatives(idf, dims, dim, sign, n_reps):
    other_dims = [d for d in dims if d != dim]
    bias_weight = 0.25
    idf['score'] = sign * idf[dim] + bias_weight * idf["bias"]
    candidates = idf \
        .sort_values(by="score", ascending=False) \
        [dims + ['Title', "movieId", "score"]] \
        .head(20)
    
    clusters = find_clusters(candidates, other_dims, n_clusters=n_reps)
    
    return clusters \
        .loc[clusters.groupby("cluster")["score"].idxmax()] \
        .sort_values(by="score", ascending=False)

def tag_correlations_df(idf):
    df = M.movies_df(idf)
    dims = [c for c in df.columns if c.startswith("q")]

    rows = []

    total_votes = float(df["imdbVotes"].sum())
    print("total votes:", total_votes)

    for type in ["Genre", "Director", "Actors", "Language", "Country", "Writer"]:
        print("--", type)
        values = list(df.head(500)[type].str.split("|").explode().dropna().unique())
        for value in values:
            print(value)
            mask = df[type].str.split("|").apply(lambda xs: value in xs if isinstance(xs, list) else False)
            has_tag = mask.astype(int)

            votes = df[mask]["imdbVotes"].sum() / total_votes
            if votes < 0.01: continue
            rows.append({
                "type": type,
                "value": value,
                "size": has_tag.mean(),
                "popularity": votes,                
                **df[dims].corrwith(has_tag).to_dict()
            })

    for type in ["age_years", "BoxOffice", "tomatoScore", "imdbVotes", "imdbRating"]:
        print("--", type)
        rows.append({
            "type": "value",
            "value": type,
            "size": 1,
            "popularity": 1,
            **df[dims].corrwith(df[type]).to_dict()
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("No script.")