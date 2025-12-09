import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ml-32m/ratings.csv")

def count_df(col):
    m_count = df.groupby(col)[["rating"]].count()
    m_count = m_count.rename({"rating": "numRatings"}, axis=1)
    m_count = m_count.sort_values(by="numRatings", ascending=False)
    return m_count.reset_index()

def main():
    n_movies = 50
    n_users = 50

    top_movies = list(df["movieId"].value_counts().head(n_movies).index)
    top_users = list(df["userId"].value_counts().head(n_users).index)
    is_top_movie = df["movieId"].isin(top_movies)
    is_top_user = df["userId"].isin(top_users)
    top_ratings = df[ is_top_movie & is_top_user ]
    
    pt = top_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="first"
    )

    sns.heatmap(pt.isna(), cbar=False)
    plt.title("Missing ratings for top " + str(n_users) + " users and top " + str(n_movies) + " movies.")
    plt.show()

if __name__ == "__main__":
    main()