# Fetch data from imdb to complete database imdb.csv

IMDB_CSV = "data/custom/imdb.csv"
NUM_ITEMS = 50
REPEAT = 2
SLEEP = 0.1

import pandas as pd
import requests
import time
import re
import os
import traceback
from omdb_key import OMDB_API_KEY

# If broken data was fetched, ignore it for the next iterations instead of fetching it again and again.
ignore_list = []

# Extend the database *REPEAT* times, each time fetching the *NUM_ITEMS* most rated movies
# among those we haven't fetched so far.
def main():
    movies_file = "data/custom/extended_movies.csv"
    if not os.path.exists(movies_file):
        print("[!] extended_movies.csv not found. Aborting.")
        return

    movies_df = pd.read_csv(movies_file)
    for k in range(REPEAT):
        run(movies_df)

# Extend the database once, skipping items that are in *ignore_list*
def run(movies_df):
    imdb_df = None
    if not os.path.exists(IMDB_CSV):
        ids = list(movies_df.head(NUM_ITEMS)['imdb_tag'])
    else:
        imdb_df = pd.read_csv(IMDB_CSV)

        is_dupe = movies_df["imdb_tag"].isin(ignore_list)
        is_treated = movies_df["imdb_tag"].isin(imdb_df["imdbID"])
        waiting_list = movies_df[~is_dupe & ~is_treated]
        ids = list(waiting_list.head(NUM_ITEMS)['imdb_tag'])
        print("===================================================================")
        print(len(imdb_df), "/", len(movies_df), "fetched (", round(100*len(imdb_df)/len(movies_df),2), "% ).")
        print("===================================================================")


    rows = []
    for id in ids:
        try:
            info = fetch_imdb_data(id)
            if info == None:
                print(" [!] No object fetched for id", id)
                ignore_list.append(id)
                break

            print(info['Title'], end=" - ", flush=True)
            if info['Title'] == None:
                print(" [!] No data fetched")
                break
            elif info['Title']=="#DUPE#":
                print("\n Broken data (#DUPE#) for id", id)
                print(movies_df[movies_df['imdb_tag']==id]['title'])
                ignore_list.append(id)
            else:
                rows.append(info)
        except Exception as e:
            print(" [!] movie:", movies_df[movies_df['imdb_tag']==id]['title'])
            print(" [!] Error:", e)
            traceback.print_exc()
            break

        # Give a small break to the API
        time.sleep(SLEEP)

    # Append new items to our database:
    new_df = pd.DataFrame(rows)
    if imdb_df is None:
        new_df.to_csv(IMDB_CSV, index=False)
    else:
        pd.concat([imdb_df, new_df], ignore_index=True).to_csv(IMDB_CSV, index=False)

def get_tomato_rating(ratings):
    if not isinstance(ratings, list):
        return None
    for rating in ratings:
        if rating.get("Source") == "Rotten Tomatoes":
            return rating.get("Value")
    return None
        
def convert_response(json):
    if json.get("Response") == "False":
        return None
    
    fields = [
        "imdbID", "Title", "Rated", "Released", "Runtime",
        "Genre", "Director", "Writer", "Actors", "Language", "Country",
        "Awards", "BoxOffice", 
        "Metascore", "imdbRating",
        "imdbVotes"
    ]

    # Gather all relevant values
    result = {}
    for field in fields:
        result[field] = json.get(field)
    result["tomatoScore"] = get_tomato_rating(json.get("Ratings"))

    # Convert comma-separated fields
    list_fields = ["Genre", "Director", "Writer", "Actors", "Language", "Country"]
    for field in list_fields:
        pipe_separated_list = re.sub(r"\s*,\s*", "|", result[field])
        result[field] = pipe_separated_list

    # Convert runtime to integer minutes (None if anything went wrong)
    runtime = result['Runtime']
    if runtime[-4:] == " min":
        try:
            result['Runtime'] = int(runtime.split()[0])
        except ValueError:
            result['Runtime'] = None
    else:
        result['Runtime'] = None

    # Make sure we don't have commas
    result['Awards'] = re.sub(",", ";", result['Awards'])

    # Money to int
    money_string = result['BoxOffice']
    if money_string != None and money_string[:1] == "$":
        result['BoxOffice'] = int(money_string.replace("$", "").replace(",", ""))
    else:
        result['BoxOffice'] = None

    # Votes to int
    try:
        result['imdbVotes'] = int(result['imdbVotes'].replace(",", ""))
    except ValueError:
        result['imdbVotes'] = None

    # Floats
    for field in ["Metascore", "imdbRating", "tomatoScore"]:
        if field in result and result[field] != None:
            m = re.search(r"[-+]?\d*\.?\d+", result[field])
            result[field] = float(m.group()) if m else None

    return result

def fetch_imdb_data(imdb_tag):
    url = f"http://www.omdbapi.com/?i={imdb_tag}&apikey={OMDB_API_KEY}&r=json"
    r = requests.get(url)
    if r.status_code != 200:
        print("FETCHING FAILED:", r.status_code)
        return None
    return convert_response(r.json())

if __name__ == "__main__":
    main()