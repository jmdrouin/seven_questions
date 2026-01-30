import streamlit as st
from src.smodel import model as Model
import plotly.express as px
from src.smodel import analyze_axes as Axes
from src.app import figures
import numpy as np
import re
import pandas as pd

@st.cache_resource
def users_df():
    bundle = Model.demo_bundle()
    return bundle['users']

@st.cache_resource
def movies_df():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=200)

@st.cache_resource
def axes_df():
    bundle = Model.demo_bundle()
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
def movies_df_large():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=1000)

@st.cache_resource
def movies_df_full():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'])

@st.cache_data
def typical_movies(idf, dim, sign):
    dims = [c for c in idf.columns if c.startswith("q")]
    return Axes.find_representatives(idf, dims, dim, sign, n_reps=4)

def control_panel(labels_left, labels_right):
    values = []
    for i in range(len(labels_left)):
        with st.expander("Question " + str(i+1)):
            colL, colS, colR = st.columns([2, 1, 2])

            with colL:
                pills(labels_left[i]["films"])
                pills(labels_left[i]["tags"], "#3535DC")

            with colR:
                pills(labels_right[i]["films"])
                pills(labels_right[i]["tags"], "#3535DC")

            with colS:
                v = st.slider(f"p{i}", -1.0, 1.0, 0.0, step=0.1, label_visibility="collapsed")
                values.append(v)

    return values

def recommendations(values):
    df = Model.recommend_items(values, movies_df_large())
    st.dataframe(df.head(20))

def pills(texts, color = "#110011"):
    style = f"display:inline-block;border:1px solid white;padding:4px 8px;margin:2px;border-radius:12px;background:{color};font-size:0.85em;"
    st.markdown(
        "".join(f"<span style='{style}'>{text}</span>" for text in texts),
        unsafe_allow_html=True
    )

def demo():
    st.write("### DEMO")
    st.write("#### What do you feel like watching?")

    labels_left = []
    labels_right = []
    idf = movies_df()
    axes = axes_df()
    dims = [c for c in idf.columns if c.startswith("q")]
    for dim in dims:
        candidates = typical_movies(idf, dim, 1)
        labels_right.append({
            "films": candidates["Title"].values,
            "tags": list(axes.sort_values(by=dim, ascending=False).head(10)['name'])
        })

        candidates = typical_movies(idf, dim, -1)
        labels_left.append({
            "films": candidates["Title"].values,
            "tags": list(axes.sort_values(by=dim, ascending=True).head(10)['name'])
        })

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
    st.divider()
    st.write("### Dimension " + str(k))

    idf["up"] = idf[dim] + idf["bias"]
    idf["down"] = -idf[dim] + idf["bias"]

    #st.dataframe(idf.sort_values(by="up",ascending=False).head(10)[['Title', dim]])

    st.write("### TYPICAL FILMS (LEFT)")
    st.dataframe(typical_movies(idf, dim, -1)['Title'])

    st.write("### TYPICAL FILMS (RIGHT)")
    st.dataframe(typical_movies(idf, dim, 1)['Title'])

    axes = axes_df()
    xdf = axes.sort_values(by=dim, ascending=True)
    st.dataframe(xdf.head(10))

    fig = figures.movie_scatter(idf, x=dim, title=dim)
    st.plotly_chart(fig)

    x=dim
    fig = px.bar(
        xdf.sort_values(by=x).tail(10),
        x=x,
        y="name",
        orientation="h",
        title="Correlation with target variable",
    )
    fig.update_layout(
        xaxis_title="Correlation",
        yaxis_title="",
    )
    st.plotly_chart(fig)

    xdf['reverse'] = -xdf[x]
    fig = px.bar(
        xdf.sort_values(by='reverse').tail(10),
        x='reverse',
        y="name",
        orientation="h",
        title="Correlation with target variable",
    )
    fig.update_layout(
        xaxis_title="Correlation",
        yaxis_title="",
    )
    st.plotly_chart(fig)

def movies():
    st.write("## Dimensions")
    idf = movies_df()
    dims = [c for c in idf.columns if c.startswith("q")]

    page = st.radio(
        "",
        options=list(range(len(dims))),   
        index=0,
        horizontal=True,
        key="page_index",
    )
    explore_dim(idf, dims, page)


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

def introduction():
    st.write("## Introduction")

    st.subheader("How recommenders usually work")
    st.markdown("""
        - **What you watched** (implicit history)  
        - **What you liked**  
        - **How you rated movies** (explicit ratings)  

        They infer a stable profile and recommend what you *usually like*.
    """)

    st.subheader("Why that fails on movie night")
    st.markdown("""
        - With my son → **scary thrillers**  
        - With a friend → **kids movies**  
        - I want to relax → **slow arthouse dramas**  

        Good movies — **wrong context**.
    """)

    st.divider()

    st.header("What if we asked about tonight instead of your past?")
    st.markdown("""
    With a handful of quick choices like...
    - **arthouse or blockbuster?**
    - **family-friendly or mature?**
    - **light or heavy?**
                
    ...you describe the group’s mood — and get recommendations immediately.
    """)

    st.divider()

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

def model_1():
    st.subheader("Rating prediction model (SVD)")
    st.latex(r"\Huge \hat{r}_{u,i} = \mu + b_i + b_u + q_i^T p_u")
    st.markdown("**Where:**")

    st.markdown(r"$\mu$: global average rating")
    st.markdown(r"$b_i$: Bias for item $i$ — is this a good movie?")
    st.markdown(r"$b_u$: Bias for item $i$ — is this a generous user?")
    st.write("SVD (Singular Value Decomposition)")
    st.markdown(r"$q_i^T p_u$: Does movie $i$ fit the taste of user $u$?")
    st.markdown(r"$q_i$: n-dimensional vector of latent factors for movie $i$")
    st.markdown(r"$p_u$: n-dimensional vector of latent factors for user $u$")

    st.divider()
    st.subheader("Algorithm (Surprise SVD)")
    st.write("Minimize:")
    st.latex(r"""
             \Huge \sum_{r_{u,i} \in R}
             (e_{u,i})^2
             + \lambda ( b_i^2 + b_u^2 + \| q_i \|^2 + \| p_u \|^2 )
    """)
    st.markdown(r"Where $e_{u,i} = r_{u,i} - \hat{r}_{u,i}$ ")
    st.markdown("by stochastic gradient descent (repeat `n_epochs` times):")
    st.latex(r"""
        \begin{aligned}
        b_u &\leftarrow b_u + \gamma (e_{ui} - \lambda b_u) \\
        b_i &\leftarrow b_i + \gamma (e_{ui} - \lambda b_i) \\
        \mathbf{p}_u &\leftarrow \mathbf{p}_u + \gamma (e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u) \\
        \mathbf{q}_i &\leftarrow \mathbf{q}_i + \gamma (e_{ui}\mathbf{p}_u - \lambda \mathbf{q}_i)
        \end{aligned}
    """)
    st.markdown("**Where:**")
    st.markdown(r"$\lambda$ is the regularization term (`reg`)")
    st.markdown(r"$\gamma$ is the learning rate (`lr`)")

    st.divider()
    st.subheader("Metrics for recommending")
    st.markdown("""
        - RMSE **Root Mean Squared Error**: How well we approximate the real rating.
        - FCP **Fraction of Concordant Pairs**: Given two films for a user, how likely it is that the model ranks them correctly.
        - NCDG@10 **How "good" is the 10-movies recommendation of the model.
    """)

    st.subheader("NCDG@k - example")
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Rank", "Weight", "Recommendation", "DCG", "Ideal", "IDCG"],
            #fill_color="lightgrey",
            align="left",
            font=dict(size=20),
            height=40
        ),
        cells=dict(
            values=[
                [1, 2, 3, ""],
                [1.00, 0.63, 0.50, ""],
                ["Alien (4★)", "Rocky (1★)", "Avatar (5★)", ""],
                [4.0, 0.63, 2.5, 7.13],
                ["Avatar (5★)", "Jaws (5★)", "Alien (4★)", ""],
                [5.0, 3.15, 2.0, 10.15]
            ],
            #fill_color="white",
            align="left",
            font=dict(size=20),
            height=40
        )
    )])
    st.plotly_chart(fig, use_container_width=True)

    st.write("NCDG@3 = DCG@3 / IDCG@3 = 7.13 / 10.15 = 0.7025")

@st.cache_resource
def models_df():
    import pandas as pd
    df = pd.read_json("explore_models_results.txt", lines=True)
    for col in ["e", "f", "pca", "lr", "reg"]:
        df[col] = df["algo"].str.extract(fr"{re.escape(col)}=([0-9]*\.?[0-9]+)").astype(float)
    df["t"] = df["algo"].str.extract(fr"t=(.*)")
    return df

def model_2():
    df = models_df()
    f_results = df[df["lr"].isna() & ~df["f"].isna() & df["pca"].isna() & df["t"].isna()]

    st.write("# Number of latent factors")
    metric = st.selectbox("metric", ["rmse_mean", "fcp_mean", "ndcg_at_10_mean"], index=0)

    fig = px.scatter(
        f_results,
        x="f",
        y=metric,
        hover_name="algo"
    )

    random_value = df[df["algo"]=="random"][metric].values[0]
    baseline_value = df[df["algo"]=="baseline"][metric].values[0]
    ideal_value = 1.0
    if metric is "rmse_mean": ideal_value = 0.0

    show_baseline = st.checkbox("Show baseline", value=False)
    show_ideal = st.checkbox("Show ideal value", value=False)
    show_random = st.checkbox("Show random baseline", value=False)

    if show_random:
        fig.add_hline(
            y=random_value,
            line_color="red",
            line_dash="dash",
            annotation_text="random",
            annotation_position="top left",
        )

    if show_baseline:
        fig.add_hline(
            y=baseline_value,
            line_color="red",
            line_dash="dash",
            annotation_text="baseline",
            annotation_position="top left",
        )
    
    if show_ideal:
        fig.add_hline(
            y=ideal_value,
            line_color="green",
            line_dash="dash",
            annotation_text="ideal value",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Number of latent factors (f)",
        yaxis_title=metric,
    )
    

    st.plotly_chart(fig)

    f50_results = df[~df["lr"].isna()]

    x_axis = st.selectbox("x_axis", ["e", "lr", "reg"], index=2)

    fig = px.scatter(
        f50_results,
        x=x_axis,
        y=metric,
        hover_name="algo"
    )
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=metric,
    )

    st.plotly_chart(fig)

def model_3():
    df = models_df()

    st.write("## Reducing dimensions with PCA")

    metric = st.selectbox("metric", ["rmse_mean", "fcp_mean", "ndcg_at_10_mean"], index=0)
    random_value = df[df["algo"]=="random"][metric].values[0]
    baseline_value = df[df["algo"]=="baseline"][metric].values[0]
    ideal_value = 1.0
    if metric is "rmse_mean": ideal_value = 0.0

    show_baseline = st.checkbox("Show baseline", value=False)
    show_ideal = st.checkbox("Show ideal value", value=False)
    show_random = st.checkbox("Show random baseline", value=False)

    pca_results = df[(~df["pca"].isna()) & (df["e"] == 40.0)]
    print(pca_results[["e"]])

    fig = px.scatter(
        pca_results,
        x="pca",
        y=metric,
        hover_name="algo"
    )
    fig.update_layout(
        xaxis_title="pca",
        yaxis_title=metric,
    )

    if show_random:
        fig.add_hline(
            y=random_value,
            line_color="red",
            line_dash="dash",
            annotation_text="random",
            annotation_position="top left",
        )

    if show_baseline:
        fig.add_hline(
            y=baseline_value,
            line_color="red",
            line_dash="dash",
            annotation_text="baseline",
            annotation_position="top left",
        )
    
    if show_ideal:
        fig.add_hline(
            y=ideal_value,
            line_color="green",
            line_dash="dash",
            annotation_text="ideal value",
            annotation_position="top left",
        )

    st.plotly_chart(fig)



def preprocessing():
    st.write("### 1. Keep 10K movies with the most ratings")

    st.write("### 2. Keep 100K users with the most divergent ratings")
    st.dataframe(users_df().head())

    mdf = movies_df_full()
    mdf = mdf \
        .set_index('movieId') \
        .drop(columns=mdf.filter(regex=r"^q\d+$").columns) \
        .drop(columns=['Unnamed: 0', 'bias', 'index'])
    
    st.dataframe(
        mdf.head(3).T,
        use_container_width=True,
        height=400
    )

    fig = px.histogram(
        mdf,
        x="ratings_count",
        nbins=50,
        title="Number of ratings per movie",
    )

    fig.update_xaxes(title="Number of ratings")
    fig.update_yaxes(title="Number of movies")

    st.plotly_chart(fig, use_container_width=True)

    st.write(mdf.describe())
