# Data:
Download the database ml-20m.zip from
https://grouplens.org/datasets/movielens/20m/
and unzip it as ./data/ml-20m/...

File structure:
movie_recoco/
├── data/
    └── ml-20m/
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