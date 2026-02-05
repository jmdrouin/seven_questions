# Allow import to reach for app module:
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports
import streamlit as st
import app.pages as st_pages
from app.demo import demo as demo_page
from app.data import data as data_page
from app.movies import movies as movies_page

# App
style = """
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
"""

st.sidebar.title("Seven Questions")

st.set_page_config(layout="wide")
st.markdown(style, unsafe_allow_html=True)

pages = {
    "Introduction": st_pages.introduction,
    "Data": data_page,
    "Model": st_pages.model_1,
    "Optimization": st_pages.model_2,
    "Reducing Dimensions": st_pages.model_3,
    "Seven Questions": movies_page,
    "Demo": demo_page,
    "Next Steps": st_pages.conclusion
}

page = st.sidebar.radio("Go to", pages.keys())
pages[page]()
