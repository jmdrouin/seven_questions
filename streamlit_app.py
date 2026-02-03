import streamlit as st
import src.app.pages as st_pages
import src.app.data as data


#st.title("Questions and Recommendations")
st.sidebar.title("Seven Questions")

st.set_page_config(layout="wide")
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
    .big-table table {
        font-size: 1.6rem;
    }
    .big-table th, .big-table td {
        padding: 0.8rem 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

pages = {
    "Introduction": st_pages.introduction,
    "Data": data.data,
    #"Preprocessing the data set": st_pages.preprocessing,
    "Model": st_pages.model_1,
    "Optimization": st_pages.model_2,
    "Reducing Dimensions": st_pages.model_3,
    "Seven Questions": st_pages.movies,
    #"Users": st_pages.users,
    "Demo": st_pages.demo,
    "Next Steps": st_pages.conclusion
}

page = st.sidebar.radio("Go to", pages.keys())
pages[page]()
