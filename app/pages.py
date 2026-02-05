import streamlit as st
import plotly.express as px
from app.components import section_title, big_divider, table, pill_box, phone_mockup, problem_solution
import app.dataframes as dataframes

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

def model_2():
    st.write("# Optimizing the prediction model")
    st.write("""
        - Average results over a 3-splits K-Fold cross-validations
        - Trained on a slice of 20k users (for speed reasons)
    """)

    df = dataframes.models()
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
    df = dataframes.models()

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