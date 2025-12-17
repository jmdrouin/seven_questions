from dataframes import top_ratings, top_movies
movies = top_movies()
ratings = top_ratings()

core_movies = []
good_users = []

print("Num ratings:", len(ratings))
prev_num_users = -1
for i in range(500):
    top_movie = ratings["movieId"][~ratings["movieId"].isin(core_movies)].mode().values[0]
    print(movies[movies["movieId"]==top_movie]["title"].values[0])
    core_movies.append(top_movie)
    good_users = ratings[ratings["movieId"]==top_movie]["userId"].unique()
    ratings = ratings[ratings["userId"].isin(good_users) & ~ratings["movieId"].isin(core_movies)]
    print("Num ratings:", len(ratings), "Num users:", len(good_users), "Num movies:", len(core_movies))
    if prev_num_users==len(good_users):
        break
    prev_num_users = len(good_users)

