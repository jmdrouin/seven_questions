from dataframes import top_movies
import pandas as pd

def explore_axes(movies_df):
    print("\n\n===== EXPLORATION OF AXES =====")

    dim_cols = [c for c in movies_df.columns if c[:4] == "dim_"]
    full_movies_df = top_movies()
    full_movies_df = full_movies_df.merge(movies_df, on="movieId")

    explore_tags(full_movies_df)
    return

    print("COLUMNS:", full_movies_df.columns)
    genres_dummies = full_movies_df["genres"].str.get_dummies(sep="|")
    movies_with_genres = pd.concat([full_movies_df, genres_dummies], axis=1)

    pg_dummies = full_movies_df["Rated"].str.get_dummies()
    movies_with_genres = pd.concat([movies_with_genres, pg_dummies], axis=1)

    print(movies_with_genres)

    genre_cols = list(genres_dummies.columns)# + list(pg_dummies.columns)

    cols = genre_cols + dim_cols
    corr = movies_with_genres[cols].corr(numeric_only=True)
    print("CORR")
    subcorr = corr.loc[genre_cols][dim_cols]
    print(subcorr)

    if False:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            subcorr,
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            annot=True
        )
        plt.title("Correlation between MF factors and genres")
        plt.show()

    if False:
        #find_points(full_movies_df, dim_cols)
        k = 10
        df = find_clusters(full_movies_df, dim_cols, n_clusters=k)
        print(df.head())

        for i in range(k):
            print("CLUSTER", i)
            dfk = df[df['cluster']==i].sort_values(by="ratings_count")
            print(dfk.tail())

        return

    if False:
        good_movies = full_movies_df.sort_values(by="ratings_count").tail(1000)
        cols = ['Title', 'Genre', 'ratings_count'] + dim_cols
        good_movies = good_movies[cols]

        good_movies["excentricity"] = (good_movies[dim_cols]**2).sum(axis=1)

        print("\n\n ---- ISO 0+ ----")
        good_movies["tmp"] = good_movies["excentricity"] - good_movies["dim_0"]**3
        print(good_movies.sort_values(by="tmp").head())

        print("\n\n ---- ISO 0- ----")
        good_movies["tmp"] = good_movies["excentricity"] + good_movies["dim_0"]**3
        print(good_movies.sort_values(by="tmp").head())

        print("\n\n ---- AVERAGE MOVIES ----")
        print(good_movies.sort_values(by="excentricity").head())

        for col in dim_cols:
            sorted = good_movies.sort_values(by=col)
            print("\n -----", col, "---------------------")
            print("BOTTOM:", list(sorted.head()['Title']))
            print("TOP:", list(sorted.tail()['Title']))

    if False:
        print("""
            ==================================
            = CLUSTERS                       =
            ==================================
        """)
        from sklearn.cluster import KMeans
        df = full_movies_df[full_movies_df["ratings_count"] > 10000]
        K = 30
        kmeans = KMeans(n_clusters=K, n_init=20, random_state=0)
        labels = kmeans.fit_predict(df[dim_cols])
        centers = kmeans.cluster_centers_
        df["cluster"] = labels

        import numpy as np
        
        for label in range(K):
            print("CLUSTER", label)
            center = centers[label]
            dfn = df[df["cluster"]==label]
            dfn["dist"] = np.linalg.norm(dfn[dim_cols] - center, axis=1)
            #dfn["score"] = (2 ** dfn["dist"]) / dfn["ratings_count"]
            print(dfn.sort_values(by="dist").head()[["Title", "dist", "ratings_count"]])


def explore_tags(df):
    df = df.copy()
    dim_cols = [c for c in df.columns if c[:4] == "dim_"]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[dim_cols] = scaler.fit_transform(df[dim_cols])

    print(df.head())
    genres = df['genres'].dropna().str.split('|').explode().unique()

    tdf = pd.DataFrame(df[dim_cols].mean()).T
    tdf["name"] = "ALL"
    tdf["count"] = len(df)

    def add_row(tdf, name, mask):
        subset = df[mask]
        if len(subset) < 10: return tdf
        mean = subset[dim_cols].mean()
        antimean = df[~mask][dim_cols].mean()
        score = mean - antimean
        row = pd.DataFrame(score).T
        row["name"] = name
        row["count"] = len(subset)
        return pd.concat([tdf, row])

    for genre in genres:
        tdf = add_row(tdf, "genre:"+genre, df["genres"].str.contains(genre))

    tags = pd.read_csv('data/ml-32m/tags.csv')
    for tag, ids in tags.groupby('tag')['movieId']:
        tdf = add_row(tdf, tag, df["movieId"].isin(ids))

    top = tdf.sort_values(by="count").tail(100)

    import plotly.express as px

    fig = px.imshow(
        top.set_index("name")[dim_cols],
        color_continuous_scale="RdBu",
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        height=2000,             # make figure tall → scrollable
        yaxis_title="",
        xaxis_title=""
    )
    fig.write_html("heatmap.html")

    print("\n\n========END============")

    #tags = {
    #    ""
    #}