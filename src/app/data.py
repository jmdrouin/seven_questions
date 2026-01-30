import streamlit as st
from src.app.pages import movies_df
from src.app.top1000 import density_plot, ratings_distribution_plot
import plotly.express as px
import pandas as pd

@st.cache_resource
def ratings_df():
    return pd.read_csv("data/ml-32m/ratings.csv", nrows=1000)

@st.cache_data
def load_aggregates():
    return (
        pd.read_csv("rating_distribution.csv"),
        pd.read_csv("ratings_per_user.csv"),
        pd.read_csv("ratings_per_movie.csv"),
    )

@st.cache_data
def load_user_means():
    return pd.read_csv("user_avg_rating.csv")

@st.cache_data
def load_movie_means():
    return pd.read_csv("movie_avg_rating.csv")

def data():
    st.write("## Data")
    st.markdown("""
        ## Dataset overview

        This project is based on the **MovieLens 32M** dataset, which contains explicit movie ratings collected over multiple years.

        Each rating is defined by:
        - a **userId**
        - a **movieId**
        - an explicit **rating** (0.5 to 5.0)
        - a **timestamp**

        In this project, we focus exclusively on explicit ratings, as they provide a clear and interpretable signal of user preference.
    """)

    ratings = ratings_df()
    st.divider()

    st.subheader("What do ratings look like?")
    st.markdown("""
    Each row corresponds to a single explicit rating given by a user to a movie.
    """)
    st.dataframe(ratings.sample(10, random_state=100), use_container_width=True)

    rating_dist, ratings_per_user, ratings_per_movie = load_aggregates()
    fig = px.bar(
        rating_dist,
        x="rating",
        y="count",
        title="Distribution of ratings"
    )
    st.plotly_chart(fig, use_container_width=True)

    users(ratings_per_user)
    movies(ratings_per_movie)

    st.divider()
    density_plot()
    st.divider()
    ratings_distribution_plot()

    st.subheader("Distribution of average ratings per user")

    st.divider()
    fig = px.histogram(load_user_means(), x="avg_rating", nbins=30, title="Average rating per user")
    fig.update_xaxes(title="User average rating")
    fig.update_yaxes(title="Number of users")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    movie_means = load_movie_means()
    fig = px.histogram(movie_means, x="avg_rating", nbins=60, title="Average rating per movie")
    fig.update_xaxes(title="Average rating")
    fig.update_yaxes(title="Number of movies")
    st.plotly_chart(fig, use_container_width=True)

    #imdb(mdf)

@st.cache_data
def load_user_signal():
    return pd.read_csv("user_signal_strength.csv")

def users(ratings):
    st.divider()
    st.markdown("""
        ## Users and activity

        The dataset contains a large number of users, but their activity levels vary significantly.
        Some users rate only a handful of movies, while others rate hundreds or thousands.                
    """)

    st.subheader("Users and ratings per user")

    @st.cache_data
    def load_ratings_per_user():
        return pd.read_csv("ratings_per_user.csv")

    ratings_per_user = load_ratings_per_user()

    n_users = len(ratings_per_user)
    st.metric("Number of users", f"{n_users:,}")

    max_ratings = st.slider(
        "Maximum number of ratings per user",
        min_value=10,
        max_value=int(ratings_per_user["n_ratings"].max()),
        value=10000,
        step=50,
    )

    filtered = ratings_per_user[ratings_per_user["n_ratings"] <= max_ratings]

    fig = px.histogram(
        filtered,
        x="n_ratings",
        nbins=50,
        log_y=True,
        title=f"Ratings per user (≤ {max_ratings}, log scale)",
    )

    st.plotly_chart(fig, use_container_width=True)

    user_signal = load_user_signal()
    st.subheader("User rating signal strength")
    fig = px.histogram(
        user_signal,
        x="min_above_below",
        nbins=50,
        title="Minimum number of ratings above or below the user’s own average",
    )
    fig.update_xaxes(title="min(#ratings above mean, #ratings below mean)")
    fig.update_yaxes(title="Number of users")
    st.plotly_chart(fig, use_container_width=True)

    """
    fig = px.histogram(
        ratings_per_user,
        x="n_ratings",
        nbins=50,
        log_y=True,
        title="Number of ratings per user (log scale)",
    )

    fig.update_xaxes(title="Ratings per user")
    fig.update_yaxes(title="Number of users")

    st.plotly_chart(fig, use_container_width=True)
    """

    st.markdown("""
        This long-tail distribution highlights why cold-start is a fundamental problem:
        many users provide very little information, making it hard to infer preferences
        from history alone.
    """)

def movies(ratings):
    st.divider()
    st.subheader("Movies and ratings per movie")

    @st.cache_data
    def load_ratings_per_movie():
        return pd.read_csv("ratings_per_movie.csv")

    ratings_per_movie = load_ratings_per_movie()

    n_movies = len(ratings_per_movie)
    st.metric("Number of movies", f"{n_movies:,}")

    max_value = int(ratings_per_movie["n_ratings"].max())
    max_ratings = st.slider(
        "Maximum number of ratings per movie",
        min_value=10,
        max_value=max_value,
        value=max_value,
        step=50,
    )

    filtered = ratings_per_movie[ratings_per_movie["n_ratings"] <= max_ratings]

    fig = px.histogram(
        filtered,
        x="n_ratings",
        nbins=50,
        log_y=True,
        title="Number of ratings per movie (log scale)",
    )

    fig.update_xaxes(title="Ratings per movie")
    fig.update_yaxes(title="Number of movies")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        As expected, movie popularity is also highly skewed:
        a small number of movies receive most of the ratings,
        while the majority are rated only a few times.

        This imbalance impacts both recommendation quality and evaluation metrics.
    """)


def imdb(mdf):
    st.divider()
    # TODO: Oops. Columns were duplicated during the merge
    mdf = mdf.rename(columns={
        "tomatoScore_y": "tomatoScore",
        "Runtime_y": "Runtime",
        "age_years_y": "age_years"
    })
    cols = ['Runtime', 'tomatoScore', 'age_years']

    for col in cols:
        st.write("### " + col)
        fig = px.histogram(mdf, col, nbins=100)
        st.plotly_chart(fig)

        st.write("### " + col + " (standardized)")
        col_std = col + "_std"
        fig = px.histogram(mdf, col_std, nbins=100)
        st.plotly_chart(fig)


    # dims = ['Runtime', 'tomatoScore', 'age_years']
    st.dataframe(mdf.head()[['tomatoScore']])
