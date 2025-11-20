import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def simplePlot():
    destination = "tmp/plot.png"
    df = pd.DataFrame({
        "x": [1,2,3,4],
        "y": [3,1,4,2]
    })

    print("Dummy data:")
    print(df)

    sns.lineplot(data=df, x="x", y="y")
    plt.title("Example Plot")
    plt.savefig(destination, dpi=200)
    plt.close()

    print("Plot saved to", destination)


def data():
    dir = "data/ml-32m/"
    file = "ratings.csv"
    print("\n")
    print("Loading", dir + file, "...")
    df = pd.read_csv( dir + file)
    print("\n===", file, " ===")
    print(df.head(20))

simplePlot()
data()
