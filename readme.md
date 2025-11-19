# Data:
Download the database ml-32m.zip from
https://grouplens.org/datasets/movielens/
and unzip it as ./data/ml-32m/...

File structure:
movie_recoco/
├── data/
    └── ml-32m/
        ├── genome-scores.csv
        ├── genome-tags.csv
        ├── links.csv
        ├── movies.csv
        ├── ratings.csv
        ├── README.txt
        └── tags.csv

# Python Environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# How to update the environment:
pip install some_library ...
pip freeze > requirements.txt