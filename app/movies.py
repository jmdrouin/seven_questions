import streamlit as st
import plotly.express as px
from app.components import section_title, big_divider, section_title
from app.plots import movie_scatter
from app.demo import control_panel
import app.dataframes as dataframes

def movies():
    section_title("Seven Questions")
    st.write("")

    st.divider()
    st.write("### A question that makes sense:")
    control_panel(
        [{"films": ['Typical Film A', 'Typical Film B', 'Typical Film C', 'Typical Film D'],
          "tags": ["Typical Genre", "Some Director", "Other indicator"]}],
        [{"films": ['Typical Film A', 'Typical Film B', 'Typical Film C', 'Typical Film D'],
          "tags": ["Typical Genre", "Some Director", "Other indicator"]}],
        compact=True
    )

    idf = dataframes.movies()
    dims = [c for c in idf.columns if c.startswith("q")]

    big_divider()
    section_title("Typical Films")
    st.write("All films $i$ can now be described with 8 numbers: $b_i, q^0_i, q^1_i, ..., q^6_i$")
    st.write("##### 200 most rated movies:")
    st.dataframe(idf[["Title", "bias"] + dims], hide_index=True)

    xdf = dataframes.axes()
    big_divider()
    section_title("Typical Tags")
    st.write("These dimensions correlate with different characteristics of the films (source: IMDB).")
    st.write("##### Genres, Actors and other tags:")
    st.dataframe(xdf[["name"] + dims + ["value", "type", "popularity"]], hide_index=True)

    big_divider()
    with st.sidebar:
        st.divider()
        st.write("## Dimension")
        page = st.radio(
            "",
            options=list(range(len(dims))),   
            index=0,
            horizontal=True,
            key="page_index",
        )

    section_title(f"Question for axis {page}")
    explore_dim(idf, dims, page)

def explore_dim(idf, dims, k):
    dim = dims[k]

    idf["up"] = idf[dim] + idf["bias"]
    idf["down"] = -idf[dim] + idf["bias"]

    typical_movies_left = dataframes.typical_movies(idf, dim, -1)
    typical_movies_right = dataframes.typical_movies(idf, dim, 1)

    fig = movie_scatter(
        idf,
        x=dim,
        y="bias",
        title=f"Top 200 films along axis {dim}"
    )
    st.plotly_chart(fig)

    st.divider()
    col0, col1, col2 = st.columns([1,2,2])
    with col1:
        st.write("#### Left side (-)")
    with col2:
        st.write("#### Right side (+)")
    st.divider()

    col0, col1, col2 = st.columns([1,2,2])
    with col0:
        st.write("#### Typical films")
        st.write(f"""(Among the top 200 films, from different clusters.)""")
    with col1:
        st.dataframe(typical_movies_left['Title'])
    with col2:
        st.dataframe(typical_movies_right['Title'])

    axes = dataframes.axes()
    xdf = axes.sort_values(by=dim, ascending=True)
    #st.dataframe(xdf.head(10))

    st.divider()
    col0, col1, col2 = st.columns([1,2,2])
    with col0:
        st.write("#### Correlated variables")
        st.write(f"""(Among IMDB variables that apply to at least 1% of the ratings.)""")
    with col2:
        x=dim
        fig = px.bar(
            xdf.sort_values(by=x).tail(10),
            x=x,
            y="name",
            orientation="h"
        )
        fig.update_layout(
            xaxis_title=f"correlation with +{dim}",
            yaxis_title="",
        )
        st.plotly_chart(fig)

    with col1:
        xdf['reverse'] = -xdf[x]
        fig = px.bar(
            xdf.sort_values(by='reverse').tail(10),
            x='reverse',
            y="name",
            orientation="h",
            title="",
        )
        fig.update_layout(
            xaxis_title=f"correlation with -{dim}",
            yaxis_title="",
        )
        st.plotly_chart(fig)