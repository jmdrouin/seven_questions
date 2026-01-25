import pickle
import pandas as pd
import numpy as np
from src.smodel import movie_info as mi

def explore_users(df):
    fig = px.histogram(df, "p3")
    for i in [5, 25, 50, 75, 95]:
        x = np.percentile(df["p3"], i)
        fig.add_vline(x=x, line_dash="solid", annotation_text=str(i)+"%")
    fig.show()

def movies_df(idf, nrows=None):
    movies_info_df = pd.read_csv("shared_data/top_movies.csv")
    movies_info_extended = mi.movie_info()[['Runtime_std', 'tomatoScore_std', 'age_years_std', 'age_years']]
    result = idf \
        .merge(movies_info_df, on="movieId") \
        .merge(movies_info_extended, on="movieId") \
        .sort_values(by="ratings_count", ascending=False)
    if nrows==None:
        return result
    else:
        return result.head(nrows)

def explore_movies(idf):
    dims = [c for c in idf.columns if c.startswith("q")]
    idf = movies_df(idf, nrows=500)

    for dim in dims:
        print("----------", dim, "---------")
        print()
        idfx = idf.sort_values(by=dim)[dims + ["Title"]]
        print(idfx.head())
        print(idfx.tail())
        print(len(idfx))

        fig = px.scatter(
            idf,
            x=dim,
            y="ratings_mean",
            hover_name="Title"
        )
        fig.update_yaxes(visible=False)
        fig.update_layout(
            title=dim,
            height=400,
        )
        
        fig.show()

    # Now let's recommend a movie...

def recommend_items(p: list[float], df: pd.DataFrame):
    dims = ["q"+str(i) for i in range(len(p))]
    df["pq_score"] = df[dims].to_numpy() @ np.array(p)
    df["rec_score"] = df["pq_score"] + df["bias"]
    sorted = df.sort_values(by="rec_score", ascending=False)
    
    return sorted[dims + ['Title', 'pq_score', 'rec_score', 'bias']]

def demo_bundle():
    with open("models/demo_svd_pca.pkl", "rb") as f:
    #with open("models/demo_svd.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle