
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
import model as Model

def main():
    bundle = Model.demo_bundle()
    
    #Model.explore_movies(bundle['items'])
    #Model.explore_users(bundle['users'])

    profiles = [
        [1,0,0,0,0],
        [1,-1,-1,-1,-1],
        [-1,0,0,0,0],
    ]
    for p in profiles:
        print()
        print("----- profile:", p)
        r = Model.recommend_items(p, Model.movies_df(bundle['items'], nrows=250))
        print(r.head())
    
if __name__ == "__main__":
    main()