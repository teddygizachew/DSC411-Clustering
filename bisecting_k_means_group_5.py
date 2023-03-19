# uses a numpy array for temp storage
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# convert categorical data to numeric
def convert_data(con_data):
    # disregard data that is already numeric
    convert_cols = con_data.select_dtypes(exclude=['int64', 'float64']).columns

    # convert potential objects to categorical
    con_data[convert_cols] = pd.Categorical(con_data[convert_cols])

    #
    con_data[convert_cols] = con_data[convert_cols].apply(lambda x: x.cat.codes)

    return con_data


# using silhouette method, determine 'k'
def silhouette_meth(sil_data):
    return k


# use bisecting kmeans method to generate clusters
def bisecting_kmeans(k, data):
    # initialize cluster list
    # list of points in cluster, centroid, Sum of Square Errors (SSE)
    clusters = [data, 0, 0]

    # cluster generation
    for i in range(0, k-1):
        # cluster with highest SSE will be last (implemented later)
        # split the last cluster in the list
        split_cluster = clusters.pop()[0]

        temp_clusters = KMeans(n_clusters=2, init="k-means++").fit(split_cluster)

        temp_clusters_labels = temp_clusters.labels_

        temp_clusters_centroids = temp_clusters.cluster_centers_

        # hold new centroids


    return clusters


# get data file name from user
filename = input("Enter the filename for clustering: ")
data = pd.read_csv(filename)

# create a second dataframe for operations to keep the original intact
bisecting_data = data

# we're only working on the first 3 columns, drop the rest
column_count = bisecting_data.columns
bisecting_data.drop(bisecting_data.iloc[:, 3:], inplace=True, axis=1)

# ensure all data is numeric, convert if not
# clean_data is also the dataframe for building the 3d model
clean_data = convert_data(bisecting_data)

#  TEMPORARY ASSIGNMENT FOR TESTING clusters from silhouette approach
# k = silhouette_meth(clean_data)
k = 4
