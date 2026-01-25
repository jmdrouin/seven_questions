import pandas as pd
import re

df = pd.read_json("explore_models_results.txt", lines=True)

for col in ["e", "f", "pca", "lr", "reg"]:
    df[col] = df["algo"].str.extract(fr"{re.escape(col)}=([0-9]*\.?[0-9]+)").astype(float)
df["t"] = df["algo"].str.extract(fr"t=(.*)")


f_results = df[df["lr"].isna() & ~df["f"].isna() & df["pca"].isna() & df["t"].isna()]
print(f_results)