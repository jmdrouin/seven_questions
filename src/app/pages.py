import streamlit as st
from src.smodel import model as Model
import plotly.express as px
from src.smodel import analyze_axes as Axes
from src.app import figures
import numpy as np

@st.cache_resource
def users_df():
    bundle = Model.demo_bundle()
    return bundle['users']

@st.cache_resource
def movies_df():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=200)

@st.cache_resource
def movies_df_large():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=1000)


@st.cache_data
def typical_movies(idf, dim, sign):
    dims = [c for c in idf.columns if c.startswith("q")]
    return Axes.find_representatives(idf, dims, dim, sign, n_reps=4)

def control_panel(labels_left, labels_right):
    st.divider()
    values = []
    for i in range(5):
        colL, colS, colR = st.columns([1, 3, 1])

        with colL:
            labels = "\n\n".join(f" {t}" for t in labels_left[i])
            st.markdown(f"<div style='text-align:right'> { labels } </div>", unsafe_allow_html=True)

        with colS:
            v = st.slider(f"p{i}", -1.0, 1.0, 0.0, step=0.1, label_visibility="collapsed")
            values.append(v)

        with colR:
            labels = "\n\n".join(f"{t}" for t in labels_right[i])
            st.markdown(labels)
        
        st.divider()

    return values

def recommendations(values):
    df = Model.recommend_items(values, movies_df_large())
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

    raw_values = control_panel(labels_left, labels_right)

    udf = users_df()
    values = []
    increment = 0.1
    for i in range(len(raw_values)):
        dim = "p" + str(i)
        percentile = (raw_values[i] + 1 + 0.5 * increment) / (2 + increment)
        mu = udf[dim].mean()
        sigma = udf[dim].std()
        #st.write(mu, sigma, percentile)
        from scipy.stats import norm
        value = norm.ppf(percentile, mu, sigma)
        values.append(value)

    st.write("#### Recommendations:")

    recommendations(raw_values)

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

    fig = figures.movie_scatter(idf, x=dim, title=dim)
    st.plotly_chart(fig)

def movies():
    st.write("### Movies")
    idf = movies_df()
    dims = [c for c in idf.columns if c.startswith("q")]
    for k in range(len(dims)):
        explore_dim(idf, dims, k)

def explore_user_dim(df, dims, k):
    dim = dims[k]

    fig = px.histogram(df, dim)
    #precision = 0.1
    #mu = df[dim].mean()
    #sigma = df[dim].std
    #from scipy.stats import norm
    #p50 = norm.ppf(0.50, mu, sigma)  # == mu

    for i in [5, 25, 50, 75, 95]:
        x = np.percentile(df["p3"], i)
        text = str(i)+"%\nx=" + str(round(x, 2))
        fig.add_vline(x=x, line_dash="solid", line_color="red", annotation_text=text)
    st.plotly_chart(fig)

def users():
    st.write("### Users")
    udf = users_df()
    dims = [c for c in udf.columns if c.startswith("p")]
    for k in range(len(dims)):
        st.write("### Dimension " + dims[k])
        explore_user_dim(udf, dims, k)

def problem():
    st.write("### Problem")

    st.write("""
        [Explain the idea]
        What should we watch tonight?
        Most systems answer this based on my past ratings and history,
        without knowing WHO I'm watching with,
        or what I'm in the mood for.
             
        Idea: Can we make a tool that assumes I'm new (cold-start user)
        and asks a few key questions on the spot. If it's quick, then I
        can start from scratch every time I'm about to watch something.        
    """)

    st.write("### Model SVD")

    st.write("Baseline:")
    st.latex(r"\Huge \hat{r}_{u,i} = \mu + b_i + b_u")
    st.markdown(r"$R_{u,i}$: Estimated rating for user $u$ and item $i$.")
    st.markdown(r"$b_i$: Bias for item i — is this a good movie?")
    st.markdown(r"$b_u$: Bias for item i — is this a generous user?")

    st.write("SVD (Singular Value Decomposition)")
    st.latex(r"\Huge \hat{r}_{u,i} = \mu + b_i + b_u + q_i^T p_u")
    st.markdown(r"$q_i^T p_u$: Does movie _i_ fit the taste of user _u_?")
    st.markdown(r"$q_i$: n-dimensional vector of latent factors for movie i")
    st.markdown(r"$p_u$: n-dimensional vector of latent factors for user i")



    #st.write(" R(u,i) = µ + b(i) + )



def data():
    st.write("### The Data")

    st.write("#### ml-32m: explicit ratings")

    st.dataframe()

    mdf = movies_df()#[dims + [d + "_std" for d in dims]]

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



