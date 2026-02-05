import streamlit as st
from src.app.pages import movies_df
from src.app.top1000 import density_plot, ratings_distribution_plot
import plotly.express as px
import pandas as pd
from src.app.figures import problem_solution, section_title, big_divider

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
    section_title("Data")

    col0, col1 = st.columns([2,4])
    with col0:
        st.markdown("""
            #### Dataset: MovieLens ml-32m
            Each row corresponds to a single explicit rating given by a user to a movie.
        """)
    with col1:
        ratings = ratings_df()
        st.dataframe(ratings.sample(5, random_state=100), use_container_width=True)
    st.divider()

    col0, col1 = st.columns([2,4])
    with col0:
        st.markdown("""
            #### IMDB data
            Each row gives us information on a film.
        """)
    with col1:
        mdf = movies_df() \
            .set_index("movieId") \
            [["Title", "Rated", "Released", "Runtime", "Genre", "Director","Writer","Actors","Language","Country","Awards","BoxOffice","Metascore","imdbRating", "imdbVotes", "tomatoScore"]]
        st.dataframe(mdf)

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

    big_divider()
    section_title("3. Sparsity of ratings")

    density_plot()
    problem_solution(
        "Ratings are very sparse.",
        "Use a model that is made for sparse ratings (scikit.surprise)."
    )

    big_divider()
    section_title("4. Different rating scales")

    fig = px.histogram(load_user_means(), x="avg_rating", nbins=30, title="Average rating per user")
    fig.update_xaxes(title="User average rating")
    fig.update_yaxes(title="Number of users")
    st.plotly_chart(fig, use_container_width=True)

    """
    movie_means = load_movie_means()
    fig = px.histogram(movie_means, x="avg_rating", nbins=60, title="Average rating per movie")
    fig.update_xaxes(title="Average rating")
    fig.update_yaxes(title="Number of movies")
    st.plotly_chart(fig, use_container_width=True)
    """

    ratings_distribution_plot()

    problem_solution(
        "Users have their own individual way to rate films. Some are harsh, some are generous.",
        "Take biases into consideration (part of the model)."
    )

    #imdb(mdf)

@st.cache_data
def load_user_signal():
    return pd.read_csv("user_signal_strength.csv")

@st.cache_data
def load_ratings_per_user():
    return pd.read_csv("ratings_per_user.csv")

def users(ratings):
    big_divider()
    section_title("1. Users")

    st.markdown("""
        The dataset contains a large number of users, but their activity levels vary significantly.
        Some users rate only a handful of movies, while others rate hundreds or thousands.                
    """)

    ratings_per_user = load_ratings_per_user()

    n_users = len(ratings_per_user)
    st.metric("Number of users", f"{n_users:,}")

    st.divider()
    graph, control = st.columns([7, 2])
    with control:
        max_ratings = st.slider(
            "Max ratings",
            min_value=10,
            max_value=int(ratings_per_user["n_ratings"].max()),
            value=10000,
            step=50,
            label_visibility="collapsed",
        )
        st.caption(f"Showing users with less than {max_ratings} ratings")
    with graph:
        filtered = ratings_per_user[ratings_per_user["n_ratings"] <= max_ratings]
        fig = px.histogram(
            filtered,
            x="n_ratings",
            nbins=50,
            log_y=False,
            title=f"Number of Ratings per user (≤ {max_ratings})",
        )
        st.plotly_chart(fig, use_container_width=True)

    user_signal = load_user_signal()
    st.divider()
    graph, control = st.columns([7, 2])
    with control:
        max_ratings_0 = st.slider(
            "Max ratings",
            min_value=10,
            max_value=int(user_signal["min_above_below"].max()),
            value=10000,
            step=50,
            label_visibility="collapsed",
        )
        st.caption(f"Showing users with less than {max_ratings_0} ratings")
    with graph:
        filtered0 = user_signal[user_signal["min_above_below"] <= max_ratings_0]
        fig = px.histogram(
            filtered0,
            x="min_above_below",
            nbins=50,
            log_y=False,
            title=f"Signal Strength — Minimum number of ratings above and below average (≤ {max_ratings})",
        )
        st.plotly_chart(fig, use_container_width=True)

    problem_solution(
        "Some users have too few ratings, or ratings that are too similar.",
        """Keep the 100k users with the most 'divergent ratings'.
        Each user has at least 28 ratings above and below their average."""
    )

@st.cache_data
def load_ratings_per_movie():
    return pd.read_csv("ratings_per_movie.csv")

def movies(ratings):
    big_divider()
    section_title("2. Movies")

    ratings_per_movie = load_ratings_per_movie()
    n_movies = len(ratings_per_movie)
    st.metric("Number of movies", f"{n_movies:,}")

    graph, control = st.columns([7, 2])
    with control:
        max_value = int(ratings_per_movie["n_ratings"].max())
        max_ratings = st.slider(
            "Maximum number of ratings per movie",
            min_value=10,
            max_value=max_value,
            value=max_value,
            step=50,
        )
    with graph:
        filtered = ratings_per_movie[ratings_per_movie["n_ratings"] <= max_ratings]
        fig = px.histogram(
            filtered,
            x="n_ratings",
            nbins=50,
            log_y=False,
            title="Number of ratings per movie",
        )
        fig.update_xaxes(title="Ratings per movie")
        fig.update_yaxes(title="Number of movies")
        st.plotly_chart(fig, use_container_width=True)

    problem_solution(
        "There are too many obscure movies with barely any rating.",
        "Keep the 10k movies with the most ratings."
    )

def imdb(mdf):
    big_divider()
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
