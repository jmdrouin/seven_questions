from dataframes import top_ratings
from split_data import split_data
from transformer import get_residuals_and_bias
from matrix_factorization import find_dimensions
import numpy as np

def main(
    test_size = None,
    num_components = None
):
    # Note: this will break if all items from a movie or a user are taken away in the test sample
    # TODO: Should we split out the most recent data instead?
    train_df, test_df = split_data(top_ratings(), test_size=test_size)

    find_model(train_df, num_components = num_components)

    return

def find_model(
    df,
    num_components=None
):
    # TODO: Should we restrict the dataset even more (min ratings per user/movie?)

    # TODO: Improve this transformation
    # TODO: This should just return a Transformer
    print(df.head())
    (residuals, user_bias, item_bias) = get_residuals_and_bias(df, n_iter=2)
    print(residuals.head())

    (item_df, user_df) = find_dimensions(residuals, user_bias, item_bias, num_components)
    print(residuals.head())
    # Note: run explore_results(movies_df) to explore item dimensions

    _ = cf_model(residuals, item_df, user_df, num_components)
    # TODO: Here, test the model using the test data.
    # However the test data doesn't have the dimensions or the biases yet.

def cf_model(residuals, item_df, user_df, num_components):
    # Clean data
    # TODO: This function should receive clean data instead of cleaning it up here.
    df = residuals \
        .drop(["user_bias", "item_bias"], axis=1) \
        .merge(item_df, on="movieId") \
        .merge(user_df, on="userId")
    
    # Target:
    y = df["residual"]

    user_cols = ["user_dim_" + str(i) for i in range(num_components)]
    item_cols = ["dim_" + str(i) for i in range(num_components)]
    A = df[user_cols].to_numpy()
    B = df[item_cols].to_numpy()
    df["dot"] = np.sum(A * B, axis=1)
    X = df[["dot"]]

    # TODO: A linear model is the easiest, not the best.
    # The linear model doesn't find an ideal coefficient of 1, which is what I'd expect,
    # but probably I understand it wrong.
    model = get_linear_model(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print("mse:", mse)
    print("score:", model.score(X, y))

    return model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
def get_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    print("\n----------LINEAR MODEL:----------")
    coef_table = pd.DataFrame({
        "feature": X.columns,
        "coef": model.coef_
    })
    print(coef_table)

    return model

if __name__ == "__main__":
    main(
        test_size = 0.2,
        num_components = 10
    )