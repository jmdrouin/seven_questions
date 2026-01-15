import streamlit as st
from src.smodel import model as Model

@st.cache_resource
def movies_df():
    bundle = Model.demo_bundle()
    return Model.movies_df(bundle['items'], nrows=1000)

def control_panel():
    labels_left  = ["Cold", "Sad", "Slow", "Simple", "Old"]
    labels_right = ["Hot", "Happy", "Fast", "Complex", "New"]
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
    st.dataframe(df.head(5))

def demo():
    st.write("### DEMO")
    st.write("#### What do you feel like watching?")
    values = control_panel()

    st.write("#### Recommendations:")
    recommendations(values)


def problem():
    st.write("### The Problem")

def data():
    st.write("### The Data")
    
    st.write("#### ml-32m: explicit ratings")

