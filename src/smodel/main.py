
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

def explore_users(df):
    fig = px.histogram(df, "p3")
    for i in [5, 25, 50, 75, 95]:
        x = np.percentile(df["p3"], i)
        fig.add_vline(x=x, line_dash="solid", annotation_text=str(i)+"%")
    fig.show()

def explore_movies(idf):
    dims = [c for c in idf.columns if c.startswith("q")]

    movies_info_df = pd.read_csv("shared_data/top_movies.csv")
    idf = idf \
        .merge(movies_info_df, on="movieId") \
        .sort_values(by="ratings_count") \
        .tail(500) # 500 most rated films
        
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

def main():
    with open("models/demo_svd.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    explore_movies(bundle['items'])
    #explore_users(bundle['users'])
    
if __name__ == "__main__":
    main()