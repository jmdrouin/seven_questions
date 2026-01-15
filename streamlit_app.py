import streamlit as st
import src.app.pages as st_pages

st.title("Questions and Recommendations")
st.sidebar.title("Table of contents")
pages = ["The Problem", "Data", "Demo"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st_pages.problem()

if page == pages[1]:
    st_pages.data()

if page == pages[2]:
    st_pages.demo()