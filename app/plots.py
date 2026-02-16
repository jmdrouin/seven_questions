import pandas as pd
import plotly.express as px
import streamlit as st
import app.dataframes as dataframes

def movie_scatter(idf, x=None, y="bias", title=None):
    fig = px.scatter(
        idf,
        x=x,
        y=y,
        hover_name="Title"
    )
    fig.update_yaxes(showticklabels=False, ticks="")
    fig.update_layout(
        title=title,
        height=400,
    )
    return fig

def top_id_block(series: pd.Series, start_rank: int, end_rank: int):
    vc = series.value_counts()
    return list(vc.iloc[start_rank - 1:end_rank].index)

def presence_block(df: pd.DataFrame, user_range=(1, 50), movie_range=(1, 50)) -> pd.DataFrame:
    users = top_id_block(df["userId"], user_range[0], user_range[1])
    movies = top_id_block(df["movieId"], movie_range[0], movie_range[1])

    block = df[df["userId"].isin(users) & df["movieId"].isin(movies)]

    pt = block.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="first",
    ).reindex(index=users, columns=movies)

    # 1 = present (YES), 0 = absent (NO)
    present = (~pt.isna()).astype(int)
    return present

def plot_presence(present: pd.DataFrame, title: str):
    fig = px.imshow(
        present.values,
        aspect="equal",  # makes it a square grid
        title=title,
        labels=dict(x="Movies", y="Users", color="Rated"),
    )

    # Force binary look: no smoothing, hide axes ticks
    fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False)

    # Make the colorbar readable (optional)
    fig.update_layout(coloraxis_colorbar=dict(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["NO", "YES"]
    ))

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig

def density_plot():
    df = dataframes.top1000ui_ratings()
    WINDOW = 50
    max_n = 1000
    max_user_start = max(1, max_n - WINDOW + 1)
    max_movie_start = max(1, max_n - WINDOW + 1)
    chart, controls = st.columns([7, 2])

    with controls:
        u_start = st.slider("User rank start", 1, max_user_start, 1, step=50)
        m_start = st.slider("Movie rank start", 1, max_movie_start, 1, step=50)
        st.caption(f"Window size: {WINDOW} × {WINDOW}")

    with chart:
        present2 = presence_block(df, user_range=(u_start, u_start + WINDOW), movie_range=(m_start, m_start + WINDOW))
        st.plotly_chart(
            plot_presence(present2, f"Users {u_start}–{u_start + WINDOW} × movies {m_start}–{m_start + WINDOW}"),
            use_container_width=True
        )


stars = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
def rating_distribution(num_items, col):
    ratings_df = dataframes.top1000ui_ratings()
    top_items = (
        ratings_df.groupby(col)["rating"]
        .count()
        .sort_values(ascending=False)
        .head(num_items)
    )

    top_items_ratings = ratings_df[ratings_df[col].isin(top_items.index)]

    ratings = (
        top_items_ratings
        .groupby([col, "rating"])
        .size()
        .unstack(fill_value=0)
    )

    ratings["num_ratings"] = top_items
    ratings = ratings.reset_index()

    #if col == "movieId":
    #    ratings = ratings.merge(movies_df, on="movieId", how="left")

    return ratings

def ratings_distribution_plot(group_col = "userId"):
    ratings = rating_distribution(25, group_col)
    # Convert counts to percentages
    ratings_percent = ratings.copy()
    ratings_percent[stars] = ratings_percent[stars].div(
        ratings_percent["num_ratings"], axis=0
    ) * 100

    id_col = "title" if group_col == "movieId" else group_col

    ratings_percent = ratings_percent.reset_index(drop=True)
    ratings_percent["row"] = ratings_percent.index

    ratings_long = ratings_percent.melt(
        id_vars=["row", "num_ratings"] + (["title"] if group_col == "movieId" else []),
        value_vars=stars,
        var_name="rating",
        value_name="percent"
    )

    colors = [
        "#460000", "#580000", "#8A0000", "#BE0000", "#FF0101", "#FF4242",
        "#5E5E5E", "#929292",
        "#00B60C", "#009D0A"
    ]

    fig = px.bar(
        ratings_long,
        x="percent",
        y="row",
        color="rating",
        orientation="h",
        color_discrete_sequence=colors,
        title="Rating distribution for top users",
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Percentage of ratings",
        yaxis_title="",
        yaxis=dict(
            showticklabels=False,   # hide meaningless numbers
            ticks=""
        ),
        height=700,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
