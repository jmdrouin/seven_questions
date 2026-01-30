import streamlit as st
import src.app.pages as st_pages
import src.app.data as data

#st.title("Questions and Recommendations")
st.sidebar.title("Seven Questions")

st.markdown("""
    <style>
    /* Increase base font size */
    html, body, [class*="css"]  {
        font-size: 20px;
    }

    /* Headers */
    h1 { font-size: 3rem; }
    h2 { font-size: 2.2rem; }
    h3 { font-size: 1.8rem; }

    /* Tables */
    thead tr th {
        font-size: 1.2rem;
    }
    tbody tr td {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

pages = {
    "Introduction": st_pages.introduction,
    "Exploration of data": data.data,
    "Preprocessing the data set": st_pages.preprocessing,
    "Model": st_pages.model_1,
    "Optimization": st_pages.model_2,
    "Dimensions": st_pages.model_3,
    "Movies": st_pages.movies,
    "Users": st_pages.users,
    "Demo": st_pages.demo
}

page = st.sidebar.radio("Go to", pages.keys())
pages[page]()
