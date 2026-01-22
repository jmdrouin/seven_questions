import plotly.express as px

def movie_scatter(idf, x=None, title=None):
    fig = px.scatter(
        idf,
        x=x,
        y="bias",
        hover_name="Title"
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        height=400,
    )
    return fig