from dataframes import top_ratings
from split_data import split_data
from transformer import BiasTransformer
from matrix_factorization import find_dimensions
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from axes import explore_axes

NUM_COMPONENTS = 10

def main(
    test_size = None,
    num_components = None
):
    # Note: this will break if all items from a movie or a user are taken away in the test sample
    # TODO: Should we split out the most recent data instead?
    # TODO: If we focus on cold-start users then we should split USERS not ratings
    train_df, test_df = split_data(top_ratings(), test_size=test_size)

    (warp, biasTransformer, model, dot) = find_model(train_df, num_components = num_components)

    t = biasTransformer.transform(warp.transform(test_df))
    t['dot'] = dot(t)

    t['bias'] = t['user_bias'] + t['item_bias'] + t['global_bias']
    t['pred_rating'] = t['bias']
    
    #residuals['calc_rating'] = warp.inverse_transform(residuals['residual'] + residuals['bias'])

    X_test = t[["dot"]]
    y_test_pred = model.predict(X_test)
    t["rough_prediction"] = warp.inverse_transform(t["bias"])
    t["full_prediction"] = warp.inverse_transform(y_test_pred + t["bias"])
    t["full_error"] = t["full_prediction"] - t["rating"]
    t["rough_error"] = t["rough_prediction"] - t["rating"]
    
    print("\n\n=============== TEST ===================")
    print(t)

    mse = mean_squared_error(t["full_prediction"], t["rating"])
    print("MSE:", mse, "RMSE:", mse ** 0.5, "MAE:", mean_absolute_error(t["full_prediction"], t["rating"]))

    for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        s = t[t["rating"]==r]
        mae0 = mean_absolute_error(s["full_prediction"], s["rating"])
        mae1 = mean_absolute_error(s["rough_prediction"], s["rating"])
        print("MAE for", str(r), "stars:", mae0, "- rough:", mae1, "- diff", mae1 - mae0)

def find_model(
    df,
    num_components=None
):
    # TODO: Should we restrict the dataset even more (min ratings per user/movie?)
    # TODO: Improve this transformation

    if False:
        warp = FunctionTransformer(
            func=lambda x: np.log2(-np.log2(1 - x / 6)),
            inverse_func=lambda x: 6.0 * (1 - 2.0 ** -(2.0 ** x))
        )
    else:
        warp = FunctionTransformer(
            func=lambda x: x,
            inverse_func=lambda x: x
        )
    
    df['original_rating'] = df['rating']
    df['rating'] = warp.transform(df['rating'])

    transformer = BiasTransformer()
    transformer.fit(df)
    residuals = transformer.transform(df)

    residuals['bias'] = residuals['user_bias'] + residuals['item_bias'] + residuals['global_bias']
    residuals['pred_rating'] = warp.inverse_transform(residuals['bias'])
    residuals['calc_rating'] = warp.inverse_transform(residuals['residual'] + residuals['bias'])

    mse = mean_squared_error(residuals['pred_rating'], residuals['original_rating'])
    print("MSE", mse)

    
    #import seaborn as sns
    #sns.histplot(residuals['residual'], bins=30, kde=True)
    #import matplotlib.pyplot as plt
    #plt.show()

    (item_df, user_df) = find_dimensions(residuals, num_components)
    explore_axes(item_df)

    X = item_dot_user(residuals, item_df, user_df, num_components)
    y = residuals["residual"]

    # TODO: A linear model is the easiest, not the best.
    # The linear model doesn't find an ideal coefficient of 1, which is what I'd expect,
    # but probably I understand it wrong.
    model = get_linear_model(X, y)
    y_pred = model.predict(X)

    residuals['final_pred_rating'] = warp.inverse_transform(residuals['bias'] + y_pred)

    print(residuals.head())

    mse = mean_squared_error(residuals['final_pred_rating'], residuals['original_rating'])
    print("mse:", mse)
    print("score:", model.score(X, y))

    def dot(df):
        return item_dot_user(df, item_df, user_df, num_components)

    return (warp, transformer, model, dot)

def item_dot_user(residuals, item_df, user_df, num_components):
    # TODO: This function should receive clean data instead of cleaning it up here.
    df = residuals \
        .drop(["user_bias", "item_bias"], axis=1) \
        .merge(item_df, on="movieId") \
        .merge(user_df, on="userId")
    user_cols = ["user_dim_" + str(i) for i in range(num_components)]
    item_cols = ["dim_" + str(i) for i in range(num_components)]
    A = df[user_cols].to_numpy()
    B = df[item_cols].to_numpy()
    df["dot"] = np.sum(A * B, axis=1)
    return df[["dot"]]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        num_components = NUM_COMPONENTS
    )