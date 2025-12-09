# Approximate user_bias, item_bias and global_bias from the given ratings dataframe.
# Model: rating(u, i) = global_bias + user_bias(u) + item_bias(i) + residual(u, i)
# Output: (residuals, user_bias, item_bias, global_bias)
def get_residuals_and_bias(df):
    global_bias = df['rating'].mean()
    df['rating_0'] = df['rating'] - global_bias
    
    # USER_BIAS:
    user_bias = df.groupby("userId")["rating_0"].mean().reset_index()
    user_bias = user_bias.rename({"rating_0": "user_bias"}, axis=1)
    df = df.merge(user_bias, on="userId", how="left")
    df["rating_1"] = df["rating_0"] - df["user_bias"]

    # FILM_BIAS
    item_bias = df.groupby("movieId")["rating_1"].mean().reset_index()
    item_bias = item_bias.rename({"rating_1": "item_bias"}, axis=1)
    df = df.merge(item_bias, on="movieId", how="left")
    df["rating_2"] = df["rating_1"] - df["user_bias"]

    # ESTIMATION OF QUALITY
    df["estimation"] = global_bias + df["user_bias"] + df["item_bias"]
    df["residual"] = df["rating"] - df["estimation"]

    residuals = df[["userId", "movieId", "residual"]]
    return (residuals, user_bias, item_bias, global_bias)
