import os
import pandas as pd

def read_or_create(name, make_dataframe):
    if os.path.exists(name):
        print("Reading", name)
        return pd.read_csv(name)
    else:
        print("Creating database", name)
        df = make_dataframe()
        df.to_csv(name)
        print("Database created:", name)
        return df
