# Script to visualise the ratings distribution for different users

import pandas as pd
movies_df = pd.read_csv("data/ml-32m/movies.csv")
ratings_df = pd.read_csv("data/ml-32m/ratings.csv")

group_col = "userId"

def rating_distribution(num_items = 5):
    top_items = ratings_df.groupby(group_col)["rating"] \
        .count() \
        .sort_values(ascending=False) \
        .head(num_items)
    top_items_ratings = ratings_df[ratings_df[group_col].isin(top_items.index)]
    ratings = top_items_ratings.groupby([group_col, "rating"]).size().unstack(fill_value=0)

    ratings = ratings.merge(top_items, on=group_col, how='left')
    ratings = ratings.rename({'rating': 'num_ratings'}, axis=1)

    if group_col=="movieId":
        # If we group by movie, fetch the movie titles
        ratings = ratings.merge(movies_df, on='movieId', how='left')

    return ratings

stars = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ratings = rating_distribution(50)
ratings_percent = ratings[stars].div(ratings['num_ratings'], axis=0) * 100

colors = [
    "#460000",
    "#580000",
    "#8A0000",
    "#BE0000",
    "#FF0101",
    "#FF4242",

    "#5E5E5E",
    "#929292",

    "#00B60C",
    "#009D0A"
]

ax = ratings_percent.plot(
    kind="barh",
    stacked=True,
    figsize=(12,5),
    edgecolor="black",
    linewidth=0.6,
    color=colors,
    width=1
)
if group_col=="movieId":
    ax.set_yticklabels(ratings['title'], rotation=0, ha='right')

import matplotlib.pyplot as plt

ax.legend(
    title="Ratings",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.subplots_adjust(left=0.4)
plt.show()