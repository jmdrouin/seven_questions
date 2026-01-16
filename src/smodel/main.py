
import pandas as pd
import pickle
import numpy as np
import model as Model

def profiles_demo(idf: pd.DataFrame):
    profiles = [
        [1,0,0,0,0],
        [1,-1,-1,-1,-1],
        [-1,0,0,0,0],
    ]
    for p in profiles:
        print()
        print("----- profile:", p)
        r = Model.recommend_items(p, Model.movies_df(idf, nrows=250))
        print(r.head())

def main():
    bundle = Model.demo_bundle()
    #profiles_demo(bundle['items'])
    Model.explore_movies(bundle['items'])
    #Model.explore_users(bundle['users'])
    
if __name__ == "__main__":
    main()