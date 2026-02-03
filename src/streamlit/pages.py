import streamlit as st
from src.smodel import model as Model
import plotly.express as px
from src.smodel import analyze_axes as Axes
from src.app import figures
import numpy as np
import re
import pandas as pd
from src.app.figures import problem_solution, section_title, big_divider, table, pill_box, phone_mockup
from scipy.stats import norm

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

def control_panel(labels_left, labels_right, compact=False):
    values = []
    for i in range(len(labels_left)):
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
        
        if compact==False:
            st.divider()
            
    if compact == False:
        bias_weight = weight_picker()
    else:
        bias_weight = 0.5

    return (values, bias_weight)

def weight_picker():
    col_left, col_slider, col_right = st.columns([1, 3, 1])

    with col_left:
        st.markdown("<div style='text-align:right; opacity:0.8;'>Just for you</div>",
                    unsafe_allow_html=True)

    with col_slider:
        value = st.slider(
            "",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("<div style='text-align:left; opacity:0.8;'>Well rated</div>",
                    unsafe_allow_html=True)
    return value

def recommendations(values, bias_weight):
    with st.sidebar:
        st.divider()
        st.write("#### 10 Recommendations:")
        df = Model.recommend_items(values, bias_weight, movies_df_large())
        st.dataframe(df.head(10)[["Title"]])


def pills(texts, color = "#110011"):
    style = f"display:inline-block;border:1px solid white;padding:4px 8px;margin:2px;border-radius:12px;background:{color};font-size:0.85em;"
    st.markdown(
        "".join(f"<span style='{style}'>{text}</span>" for text in texts),
        unsafe_allow_html=True
    )

def demo():
    section_title("What do you feel like watching?")

    st.write("")
    st.divider()


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

    raw_values, bias_weight = control_panel(labels_left, labels_right)

    # Standardize values in terms of user percentiles
    udf = users_df()
    values = []
    increment = 0.1
    for i in range(len(raw_values)):
        dim = "p" + str(i)
        percentile = (raw_values[i] + 1 + 0.5 * increment) / (2 + increment)
        mu = udf[dim].mean()
        sigma = udf[dim].std()
        value = norm.ppf(percentile, mu, sigma)
        values.append(value)

    recommendations(values, bias_weight)

def explore_dim(idf, dims, k):
    dim = dims[k]
    #st.divider()
    #st.write("### Dimension " + str(k))

    idf["up"] = idf[dim] + idf["bias"]
    idf["down"] = -idf[dim] + idf["bias"]

    #st.dataframe(idf.sort_values(by="up",ascending=False).head(10)[['Title', dim]])

    #axis = st.selectbox(
    #    "y-axis",
    #    options=["bias"] + [f"q{i}" for i in range(7)],
    #    label_visibility="collapsed",
    #)
    axis = "bias"
    fig = figures.movie_scatter(idf, x=dim, y=axis, title=f"Top 200 films along axis {dim}")
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
        st.dataframe(typical_movies(idf, dim, -1)['Title'])
    with col2:
        st.dataframe(typical_movies(idf, dim, 1)['Title'])

    axes = axes_df()
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

    idf = movies_df()
    dims = [c for c in idf.columns if c.startswith("q")]

    big_divider()
    section_title("Typical Films")
    st.write("All films $i$ can now be described with 8 numbers: $b_i, q^0_i, q^1_i, ..., q^6_i$")
    st.write("##### 200 most rated movies:")
    st.dataframe(idf[["Title", "bias"] + dims], hide_index=True)

    xdf = axes_df()
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
    section_title("Introduction")

    st.write("")
    st.write("### Traditional recommender systems work like this:")

    _, col_left, col_arrow, col_right, _ = st.columns([1, 5, 1.2, 5, 1])
    with col_left:
        pill_box(
            "Because you've watched/liked/rated...",
            ["The Godfather", "Toy Story", "Titanic", "Alien"],
            accent="#6F9FD9",
        )
    with col_arrow:
        st.markdown(
            """
            <div style="
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 3rem;
                color: #aaa;
                ">
                →
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        pill_box(
            "...you might also like:",
            ["Aliens", "Chicken Run", "2001 A Space Odyssey"],
            accent="#4CAF50",
        )
        
    st.write("")
    st.markdown(f"### However...")
    st.markdown("""
        - When I watched **Toy Story** I was with my child
        - When I watched **Alien** I was definitely not
        - When I watched **Godfather** I was in the mood for something slower.
        - Tonight I'm in the mood for something relaxing
    """)

    big_divider()

    col0, col1,_ = st.columns([2,3,2])
    with col0:
        section_title("The Idea")
        st.write("""
            With a few questions, let me tell you what I'm in the mood for right now.
            I can change my answers at any time.
        """)
    with col1:
        phone_mockup()

def model_1():
    section_title("Rating prediction model — SVD")
    st.write("`scikit.surprise.prediction_algorithms.matrix_factorization.SVD`")
    st.latex(r"\Huge \hat{r}_{u,i} = \mu + b_i + b_u + \mathbf{p}_u \cdot \mathbf{q}_i")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1: Baseline")
        st.markdown(r"$\hat{r}_{u,i}$: predicted rating for user $u$ and movie $i$")
        st.markdown(r"$\mu$: global average rating")
        st.markdown(r"$b_i$: Bias for item $i$ — is this a good movie?")
        st.markdown(r"$b_u$: Bias for user $u$ — is this a generous user?")
    with col2:
        st.markdown("##### 2: SVD (Singular Value Decomposition)")
        st.markdown(r"$\mathbf{p}_u \cdot \mathbf{q}_i$: Does movie $i$ fit the taste of user $u$?")
        st.markdown(r"$\mathbf{q}_i$: n-dimensional vector of latent factors for movie $i$")
        st.markdown(r"$\mathbf{p}_u$: n-dimensional vector of latent factors for user $u$")

    big_divider()
    section_title("Algorithm")
    st.write("Minimize:")
    st.latex(r"""
             \Large \sum_{r_{u,i} \in R}
             (e_{u,i})^2
             + \lambda ( b_i^2 + b_u^2 + \| q_i \|^2 + \| p_u \|^2 )
    """)
    st.markdown(r"Where $e_{u,i} = r_{u,i} - \hat{r}_{u,i}$ ")
    st.markdown("by stochastic gradient descent (repeat for all ratings `n_epochs` times):")
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

    big_divider()
    section_title("Metrics")

    table("""
        |Metric|Name|Measures|
        |------|----|------|
        |RMSE|Root Mean Squared Error|How well we predict the rating|
        |FCP|Fraction of Concordant Pairs|How often we predict $r_{u,i} > r_{u,j}$ correctly|
        |NCDG@10|Normalized Cumulative Discounted Gain|How good is the top-10 recommendation compared to the ideal one.|
    """)

    st.write("#### NCDG@k: Normalized Cumulative Discounted Gain")
    st.markdown("""
        | Rank | Weight | Recommendation | DCG | Ideal recommendation | Ideal DCG |
        |------|--------|----------------|-----|----------------------|-----------|
        |1     |  1×    | 1 Alien (4★)   | 4.0 |    Avatar (5★)       |  5.0      |
        |2     |  0.63× | Rocky (1★)     | 0.63|      Jaws (5★)       |  3.15     |
        |3     |  0.5×  | Avatar (5★)    | 2.5 |   Alien (4★)         |  2.0      |
        |      |        |                |     |                      |           |
        |3     |        |                |7.13 |                      |  10.15    |
    """)
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
    st.write("# Optimizing the prediction model")
    st.write("""
        - Average results over a 3-splits K-Fold cross-validations
        - Trained on a slice of 20k users (for speed reasons)
    """)

    df = models_df()
    f_results = df[df["lr"].isna() & ~df["f"].isna() & df["pca"].isna() & df["t"].isna()]

    section_title("1. Number of latent factors")
    st.write("`n_epochs=20 lr=0.005 reg=0.02`")

    with st.sidebar:
        st.divider()
        st.write("## Options")
        metric = st.selectbox("metric", ["rmse_mean", "fcp_mean", "ndcg_at_10_mean"], index=0)
        show_baseline = st.checkbox("Show baseline", value=False)
        show_ideal = st.checkbox("Show ideal value", value=False)
        show_random = st.checkbox("Show random baseline", value=False)
        random_value = df[df["algo"]=="random"][metric].values[0]
        baseline_value = df[df["algo"]=="baseline"][metric].values[0]
        ideal_value = 1.0
        if metric is "rmse_mean": ideal_value = 0.0

    fig = px.scatter(
        f_results,
        x="f",
        y=metric,
        hover_name="algo"
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

    fig.update_layout(
        xaxis_title="Number of latent factors",
        yaxis_title=metric,
    )
    
    st.plotly_chart(fig)

    big_divider()
    section_title("2. Epochs, Learning Rate, Regularization term")
    st.write("`n_factors=50`")

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

    big_divider()
    section_title("Selected Model")
    st.markdown(r"""
        - Regularization term $\lambda=0.06$
        - Learning Rate $\gamma=0.007$
        - n_epochs: 40
        - number of factors: 50
    """)

    phone_mockup()

    problem_solution(
        """The model needs 50 dimensions to describe a user.
           Do we really need to ask 50 questions?""",
        "Extract the most important dimensions without losing too much information."
    )

def model_3():
    df = models_df()

    section_title("Reducing dimensions with PCA")
    st.write("(Principal Component Analysis)")

    st.markdown("""
        To reduce the problem from 50 to m dimensions (m small),
        we find a m x 50 PCA matrix (reducing the number of variables while retaining maximum information).
        Then our model becomes:
    """)

    st.write("##### Full model (50 dimensions):")
    st.latex(r"""
        \Large \hat r_{u,i} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i
    """)

    st.write("##### Reduced model ($m$ dimensions):")
    st.latex(r"""
        \Large \hat R_{u,i} = b_u + \mathbf{z}_u \cdot ( \mathbf{W} \mathbf{q}_i )
    """)

    table("""
        | Full model | Reduced model |
        |------|----|
        | $\mu$, $b_u$ | Ignored (irrelevant for ranking) |
        | $b_i$ | Stays the same |
        | $\mathbf p_u$ (size 50) | $\mathbf z_u$ (A simplified user profile of length $m$) |
        | $\mathbf q_i$ (size 50) | $\mathbf{W} \mathbf{q}_i$ with the $50 × m$ projection matrix $W$ |
    """)

    with st.sidebar:
        st.divider()
        st.write("## Options")
        metric = st.selectbox("metric", ["fcp_mean", "ndcg_at_10_mean"], index=0)
        random_value = df[df["algo"]=="random"][metric].values[0]
        baseline_value = df[df["algo"]=="baseline"][metric].values[0]
        ideal_value = 1.0
        if metric is "rmse_mean": ideal_value = 0.0
        show_baseline = st.checkbox("Show baseline", value=False)
        show_ideal = st.checkbox("Show ideal value", value=False)
        show_random = st.checkbox("Show random baseline", value=False)

    pca_results = df[(~df["pca"].isna()) & (df["e"] == 40.0)]
    print(pca_results[["e"]])

    big_divider()
    section_title("Performance of reduced model")

    st.write("""
        **Assume** that our users will answer our questions in a way that gives the taste profile 
        that the complete model would have expected from them: $\mathbf z_u = W \mathbf p_u$.
             
        **Then** our method and metrics are still a good approximation of the results.
    """)

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

    big_divider()
    section_title("Conclusion")
    st.write("""
        It's a tradeoff. Fewer dimensions is more convenient to use, more dimensions gives more precise results.
        We keep 7 dimensions for this demo.
    """)

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

def conclusion():
    st.write("# Next Steps...")
    section_title("Improve the model")
    st.write("""
        - Combine with CBF (contents based filtering)
        - Involve implicit ratings (using SVD++ for instance)
        - Transform ratings data (highlight high ratings, avoid extreme predictions)
        - Modelize or collect data for "tonight-only" ratings
    """)

    section_title("Make it part of a hybrid system")
    st.write("""
        - Use users past history and make it part of the model
        - Add explicit questions (contents based)
        - Allow more ways to explore manually (click on a film or a tag, filter...)
        - Improve interface — questions must be simple.
    """)