# Script to display basic stats and first lines of the project's movie databases.

# File to write the result (optional)
OUTPUT = "tmp/basics.txt"

import pandas as pd

# Write to terminal and to the specified output file
def write(*args, **kwargs):
    print(*args, **kwargs)
    if OUTPUT != None:
        with open("tmp/basics.txt", "a") as f:
            print(*args, **kwargs, file=f)

# Write basic stats for the specified dataframe
def explore_df(name):
    try:
        df = pd.read_csv(name)
    except Exception as e:
        write("[!] Database", name, "could not be read.")
        return
    
    write()
    write("-" * 60)
    write("Database", name)
    write("-" * 60)
    write(df.head())
    write()
    write(df.describe())

def main():
    open("tmp/basics.txt", "w").close()

    explore_df("data/ml-20m/movies.csv")
    explore_df("data/ml-20m/ratings.csv")
    explore_df("data/ml-20m/tags.csv")
    explore_df("data/ml-20m/links.csv")
    explore_df("data/ml-20m/genome-scores.csv")
    explore_df("data/ml-20m/genome-tags.csv")

    explore_df("data/ml-32m/movies.csv")
    explore_df("data/ml-32m/ratings.csv")
    explore_df("data/ml-32m/tags.csv")
    explore_df("data/ml-32m/links.csv")

if __name__ == "__main__":
    main()