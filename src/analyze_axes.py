import pandas as pd
from src.model import movies_df
from sklearn.cluster import KMeans

def find_clusters(df, dims, n_clusters):
    """
    Split the points df[dims] into n clusters using KMeans
    
    :param df: source dataframe
    :param dims: columns considered for clustering
    :param n_clusters: Number of clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(df[dims])
    df["cluster"] = labels
    return df

def find_representatives(idf, dims, dim, sign, n_reps):
    """
    Find a set of n items that represents well one end of dimension dim.
    
    :param idf: Items dataframe
    :param dims: Rows considered in the choice
    :param dim: Dimension for which representatives are needed
    :param sign: Which end of the dimension (-1 or 1) is targeted
    :param n_reps: Number of representatives in the set.
    """

    other_dims = [d for d in dims if d != dim]
    bias_weight = 0.25
    idf['score'] = sign * idf[dim] + bias_weight * idf["bias"]
    candidates = idf \
        .sort_values(by="score", ascending=False) \
        [dims + ['Title', "movieId", "score", "bias"]] \
        .head(20)    
    clusters = find_clusters(candidates, other_dims, n_clusters=n_reps)
    return clusters \
        .loc[clusters.groupby("cluster")["score"].idxmax()] \
        .sort_values(by="score", ascending=False)

def tag_correlations_df(idf):
    """
    Build a dataframe of tags (characteristics of movies),
    indicating the correlation between the tag and the "qi" axis.
    
    :param idf: Items (movies) dataframe
    """
    df = movies_df(idf)
    dims = [c for c in df.columns if c.startswith("q")]
    rows = []
    total_votes = float(df["imdbVotes"].sum())

    for type in ["Genre", "Director", "Actors", "Language", "Country", "Writer"]:
        values = list(df.head(500)[type].str.split("|").explode().dropna().unique())
        for value in values:
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