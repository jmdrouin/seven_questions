import pandas as pd
import numpy as np

df = pd.read_csv("data/ml-32m/ratings.csv")
stats = df.groupby("userId")["rating"].agg(["mean", "std", "count"])
stats = stats[stats["count"] < 1500]
stats = stats[stats["count"] > 20]

print(stats)

import matplotlib.pyplot as plt

x = stats["count"]
y = stats["mean"]
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 300)
ys = m * xs + b

plt.scatter(stats["count"], stats["mean"],s=1, alpha=0.05)
plt.plot(xs, ys, linewidth=1, alpha=1, color='r')

plt.xlabel("Number of ratings")
plt.ylabel("Mean rating")
plt.title("Mean vs Count per userId")
plt.show()