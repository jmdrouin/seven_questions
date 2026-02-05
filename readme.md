# Data:
Download the database ml-32m.zip from
https://grouplens.org/datasets/movielens/
and unzip it as ./data/ml-32m/...

# Python Environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# How to update the environment:
pip install some_library ...
pip freeze > requirements.txt

# To start the streamlit app:
streamlit run app/main.py

# Scripts:

## explore_models.py
Model optimization.
Train models defined in the test plan and save the resulting metrics in models/explore_models_results.txt

## fetch_imdb_data.py
Calls the OMDB API to continue feeding the local movie database (data/processed/imdb.csv)

## make_demo_model.py
Trains a ratings prediction model and save relevant data as a pickle archive.

## prepare_data.py
Creates the csv files that are too big for git:
- extended_movies.csv
- top_movies.csv
- top_ratings.csv

## process_app_data.py
Prepares the dataframes needed to run the streamlit app.