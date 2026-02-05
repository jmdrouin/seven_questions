import streamlit as st
import pandas as pd
import pickle
import src.smodel.model as Model
import src.smodel.analyze_axes as Axes

def read_csv(filename):
    return pd.read_csv("app/preprocessed_data/" + filename)

def demo_bundle():
    with open("models/demo_svd_pca.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

@st.cache_resource
def ratings():
    return read_csv("ratings1000.csv")

@st.cache_data
def aggregates():
    return (
        read_csv("rating_distribution.csv"),
        read_csv("ratings_per_user.csv"),
        read_csv("ratings_per_movie.csv"),
    )

@st.cache_data
def user_means():
    return read_csv("user_avg_rating.csv")

@st.cache_data
def movie_means():
    return read_csv("movie_avg_rating.csv")

@st.cache_data
def user_signal():
    return read_csv("user_signal_strength.csv")

@st.cache_data
def ratings_per_user():
    return read_csv("ratings_per_user.csv")

@st.cache_data
def ratings_per_movie():
    return read_csv("ratings_per_movie.csv")

@st.cache_resource
def users():
    return demo_bundle()['users']

@st.cache_resource
def movies():
    bundle = demo_bundle()
    return Model.movies_df(bundle['items'], nrows=200)

@st.cache_resource
def axes():
    bundle = demo_bundle()
    df = bundle["axes"]

    VALUE_LABELS = {
        "tomatoScore": "Critically acclaimed",
        "imdbRating": "Popular",
        "BoxOffice": "Blockbuster",
        "age_years": "Classic",
        "imdbVotes": "Mainstream",
        "United States": "American"
    }

    def format_categorical(feature, value):
        if feature == "Genre":
            return str(value)
        if feature == "Director":
            return f"Directed by {value}"
        if feature == "Writer":
            return f"Written by {value}"
        if feature == "Actors":
            return f"Starring {value}"
        if feature == "Country":
            return f"From {value}"
        if feature == "Language":
            return f"In {value}"
        return f"{feature}: {value}"
    
    def make_name(row):
        feature = row["type"]
        value = row["value"]
        # Case 1: special numeric features
        if feature == "value" and value in VALUE_LABELS:
            return VALUE_LABELS[value]
        # Case 2: categorical features
        return format_categorical(feature, value)
    
    df["name"] = df.apply(make_name, axis=1)
    return df

@st.cache_resource
def movies_large():
    bundle = demo_bundle()
    return Model.movies_df(bundle['items'], nrows=1000)

@st.cache_resource
def movies_full():
    bundle = demo_bundle()
    return Model.movies_df(bundle['items'])

@st.cache_data
def typical_movies(idf, dim, sign):
    dims = [c for c in idf.columns if c.startswith("q")]
    return Axes.find_representatives(idf, dims, dim, sign, n_reps=4)

@st.cache_data
def top1000ui_ratings():
    return read_csv("top_1000ui_ratings.csv")