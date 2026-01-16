from src.smodel import model as M
import numpy as np

from sklearn.cluster import KMeans

def find_clusters(df, dims, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(df[dims])
    centers = kmeans.cluster_centers_
    df["cluster"] = labels
    return df

def find_representatives(idf, dims, dim, sign, n_reps):
    other_dims = [d for d in dims if d != dim]
    bias_weight = 0.5
    idf['score'] = sign * idf[dim] + bias_weight * idf["bias"]
    candidates = idf \
        .sort_values(by="score", ascending=False) \
        [dims + ['Title', "movieId", "score"]] \
        .head(20)
    
    clusters = find_clusters(candidates, other_dims, n_clusters=n_reps)
    
    return clusters \
        .loc[clusters.groupby("cluster")["score"].idxmax()] \
        .sort_values(by="score", ascending=False)

def main():
    bundle = M.demo_bundle()
    idf = M.movies_df(bundle["items"])
    dims = [c for c in idf.columns if c.startswith("q")]

    for dim in dims:
        for sign in [-1,1]:
            print("")
            print("--------", str(sign), str(dim), "----------")
            r = find_representatives(idf, dims, dim, sign, n_reps=4)
            print(r)

    #for dim in dims:
    #    print()
    #    print("ORDER BY", dim)
    #    print(candidates.sort_values(by=dim))

    #qs = [q for q in dims if not q == dim]
    #y = candidates[qs].iloc[0].to_numpy()
    #worst_dim = 

    #idf['dist_to_0'] = np.linalg.norm(idf[qs] - y, axis=1)

    #idf['new_score'] = 10 * idf['score'] + idf['dist_to_0']

    #print(idf.sort_values(by="new_score", ascending=False).head()[dims + ['Title', 'dist_to_0', 'score']])



    return
    dims = [c for c in idf.columns if c.startswith("q")]
    for k in range(len(dims)):
        explore_dim(idf, dims, k)

    dim = dims[k]
    st.write("### Dimension " + str(k))

    idf["up"] = idf[dim] + idf["bias"]
    idf["down"] = -idf[dim] + idf["bias"]

    st.dataframe(idf \
            .sort_values(by="up", ascending=False)[["Title", dims[k], "bias"]] \
            .head()
    )
        
    st.dataframe(idf \
            .sort_values(by="down", ascending=False)[["Title", dims[k], "bias"]] \
            .head()
    )
    
    fig = px.scatter(
        idf,
        x=dim,
        y="bias",
        hover_name="Title"
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=dim,
        height=400,
    )
    
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()