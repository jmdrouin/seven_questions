import streamlit as st
from src.smodel import model as Model
import plotly.express as px
from src.smodel import analyze_axes as Axes

@st.cache_resource
def movies_df():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=200)

@st.cache_data
def typical_movies(idf, dim, sign):
    dims = [c for c in idf.columns if c.startswith("q")]
    return Axes.find_representatives(idf, dims, dim, sign, n_reps=4)

def control_panel(labels_left, labels_right):
    values = []
    for i in range(5):
        colL, colS, colR = st.columns([1, 3, 1])

        with colL:
            st.markdown(f"<div style='text-align:right'>{labels_left[i]}</div>", unsafe_allow_html=True)

        with colS:
            v = st.slider(f"p{i}", -1.0, 1.0, 0.0, step=0.1, label_visibility="collapsed")
            values.append(v)

        with colR:
            st.markdown(labels_right[i])

    return values

def recommendations(values):
    st.write(values)
    df = Model.recommend_items(values, movies_df())
    st.dataframe(df.head(20))

def demo():
    st.write("### DEMO")
    st.write("#### What do you feel like watching?")

    #labels_left  = ["Dr. Strangelove", "The Lord of the Rings", "Harry Potter", "Eternal Sunshine of the Spotless Mind", "Forrest Gump"]
    #labels_right = ["Gladiator", "Alien", "Pulp Fiction", "The Lord of the Rings", "Casablanca"]

    labels_left = []
    labels_right = []
    idf = movies_df()
    dims = [c for c in idf.columns if c.startswith("q")]
    for dim in dims:
        candidates = typical_movies(idf, dim, 1)
        labels_right.append(candidates["Title"].values)
        candidates = typical_movies(idf, dim, -1)
        labels_left.append(candidates["Title"].values)

    values = control_panel(labels_left, labels_right)

    st.write("#### Recommendations:")
    recommendations(values)

def explore_dim(idf, dims, k):
    dim = dims[k]
    st.write("### Dimension " + str(k))

    idf["up"] = idf[dim] + idf["bias"]
    idf["down"] = -idf[dim] + idf["bias"]

    st.dataframe(idf \
            .sort_values(by="up", ascending=False)[["Title", dims[k], "bias"]] \
            .head()
    )
        
    st.dataframe(idf \
            .sort_values(by="down", ascending=False)[["Title", dims[k], "bias"]] \
            .head()
    )

    st.write("### TYPICAL FILMS (UP)")
    st.dataframe(typical_movies(idf, dim, 1))
    st.write("### TYPICAL FILMS (DOWN)")
    st.dataframe(typical_movies(idf, dim, -1))

    fig = px.scatter(
        idf,
        x=dim,
        y="bias",
        hover_name="Title"
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=dim,
        height=400,
    )
    
    st.plotly_chart(fig)


def movies():
    st.write("### Movies")

    idf = movies_df()
    dims = [c for c in idf.columns if c.startswith("q")]
    for k in range(len(dims)):
        explore_dim(idf, dims, k)


def problem():
    st.write("### The Problem")

def data():
    st.write("### The Data")
    
    st.write("#### ml-32m: explicit ratings")

