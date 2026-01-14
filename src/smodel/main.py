
import pandas as pd

from surprise import SVD
from surprise.model_selection import cross_validate

def main():
    df = pd.read_csv("shared_data/ratings_normal_users.csv")

    print(df.describe())
    print(df.head())


    
if __name__ == "__main__":
    main()