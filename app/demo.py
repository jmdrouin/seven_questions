import streamlit as st
from app.components import section_title, pills
import app.dataframes as dataframes
from scipy.stats import norm
from src.model import recommend_items

def demo():
    section_title("What do you feel like watching?")

    st.write("")
    st.divider()

    labels_left = []
    labels_right = []
    idf = dataframes.movies()
    axes = dataframes.axes()
    dims = [c for c in idf.columns if c.startswith("q")]
    for dim in dims:
        candidates = dataframes.typical_movies(idf, dim, 1)
        labels_right.append({
            "films": candidates["Title"].values,
            "tags": list(axes.sort_values(by=dim, ascending=False).head(10)['name'])
        })

        candidates = dataframes.typical_movies(idf, dim, -1)
        labels_left.append({
            "films": candidates["Title"].values,
            "tags": list(axes.sort_values(by=dim, ascending=True).head(10)['name'])
        })

    raw_values, bias_weight = control_panel(labels_left, labels_right)

    # Standardize values in terms of user percentiles
    udf = dataframes.users()
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

def recommendations(values, bias_weight):
    with st.sidebar:
        st.divider()
        st.write("#### 10 Recommendations:")
        df = recommend_items(values, bias_weight, dataframes.movies_large())
        st.dataframe(df.head(10)[["Title"]])

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
