
import pandas as pd

from surprise import SVD, accuracy, Dataset, Reader
#from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

def main():
    # WARNING: I'm using the small set of ratings here
    df = pd.read_csv("data/ratings_normal_users_small.csv")

    print(df.describe())
    print(df.head())

    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(
        df[["userId", "movieId", "rating"]],
        reader
    )

    trainset, testset = train_test_split(data, test_size=0.2)


    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)

    x = accuracy.fcp(predictions)
    print(x)

    
if __name__ == "__main__":
    main()