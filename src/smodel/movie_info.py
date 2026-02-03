from src.smodel import prepare_data
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def encode_age(df):
    df["Released"] = pd.to_datetime(
        df["Released"],
        format="%d %b %Y",
        errors="coerce"
    )
    ref_date = pd.Timestamp("2026-01-01")
    df["age_years"] = (ref_date - df["Released"]).dt.days / 365.25
    return df.drop(["Released"], axis=1)

def movie_info():
    df = prepare_data.top_movies() \
        .set_index('movieId') \
        [['Title', 'Runtime', 'tomatoScore', 'Released']]
    df = encode_age(df)
    for col in ['Runtime', 'tomatoScore', 'age_years']:
        scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=1000,
            random_state=100
        )
        scaler.fit(df[[col]])
        df[col + '_std'] = scaler.transform(df[[col]])
    return df

if __name__ == "__main__":
    df = movie_info()
    print(df.columns)
    print(df.head())