import numpy as np
import pandas as pd


# convert categorical data to numeric
def convert_data(con_data):
    # disregard data that is already numeric
    convert_cols = con_data.select_dtypes(exclude=['int64', 'float64']).columns

    # convert potential objects to categorical
    con_data[convert_cols] = con_data[convert_cols].astype('category')
    con_data[convert_cols] = con_data[convert_cols].apply(lambda x: x.cat.codes)

    # make sure all columns are floats (they're easier to work with)
    con_data = con_data.astype('float')

    return con_data


# using silhouette method, determine 'k', written by Olivia Ryan
def silhouette_meth(file):
    import csv
    # Read data from CSV file
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header
        data = []
        for row in reader:
            data.append([float(x) for x in row[0].split(',')])

    # Convert data to numpy array
    data = np.array(data)

    # Define function to calculate Euclidean distance
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Define function to calculate silhouette coefficient for a single data point
    def silhouette_coefficient(data, labels, i):
        cluster_i = labels[i]
        a_i = np.mean(
            [euclidean_distance(data[i], data[j]) for j in range(len(data)) if labels[j] == cluster_i and j != i])
        b_i = min(
            [np.mean([euclidean_distance(data[i], data[j]) for j in range(len(data)) if labels[j] == cluster_j]) for
             cluster_j in set(labels) if cluster_j != cluster_i])
        if np.isnan(a_i) or np.isnan(b_i):
            return 0
        else:
            return (b_i - a_i) / max(a_i, b_i)

    # Define function to calculate average silhouette coefficient for all data points
    def average_silhouette_coefficient(data, labels):
        return np.mean([silhouette_coefficient(data, labels, i) for i in range(len(data))])

    # Define function to perform K-means clustering
    def k_means_clustering(data, k):
        centroids = data[np.random.choice(range(len(data)), k, replace=False)]
        labels = np.zeros(len(data))
        while True:
            new_labels = np.array(
                [np.argmin([euclidean_distance(x, centroid) for centroid in centroids]) for x in data])
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for i in range(k):
                if len(data[labels == i]) > 0:
                    centroids[i] = np.mean(data[labels == i], axis=0)
        return labels

    # Define range of k values to test
    k_values = range(2, 10)

    # Calculate average silhouette coefficient for each k value
    silhouette_coefficients = []
    for k in k_values:
        labels = k_means_clustering(data, k)
        sc = average_silhouette_coefficient(data, labels)
        silhouette_coefficients.append(sc)

    optimal_k = k_values[np.argmax(silhouette_coefficients)]

    return optimal_k


# used to calculate Sum of Squared Errors
def calc_SSE(curr_data, centroid):
    sse = 0
    for a in range(0, len(curr_data)):
        eu_dist = np.sqrt(
            np.square(curr_data[a][0] - centroid[0]) + np.square(curr_data[a][1] - centroid[1]) + np.square(
                curr_data[a][2] - centroid[2]))
        sse = sse + eu_dist
    return sse


# used to assign data points to nearest centroid
def assign_centroid(curr_data, centroids):
    labels = []
    for j in range(0, len(curr_data)):
        eu_dist1 = np.sqrt(
            np.square(curr_data[j][0] - centroids[0][0]) + np.square(curr_data[j][1] - centroids[0][1]) + np.square(
                curr_data[j][2] - centroids[0][2]))
        eu_dist2 = np.sqrt(
            np.square(curr_data[j][0] - centroids[1][0]) + np.square(curr_data[j][1] - centroids[1][1]) + np.square(
                curr_data[j][2] - centroids[1][2]))

        if eu_dist1 <= eu_dist2:
            labels.append(0)
        else:
            labels.append(1)
    return labels


# used to create initial centroids
def init_centroids(curr_data, k):
    centroids = []
    for i in range(k):
        centroid = curr_data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


# used to obtain labels for the simple_KMeans method
def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


# used to create test centroids in the simple_KMeans method
def new_centroids(curr_data, labels, k):
    new_Centroids = curr_data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return new_Centroids


# used for getting next pair of centroids
def simple_KMeans(curr_data):
    centroids = []
    max_iterations = 10

    pd_DataFrame_data = pd.DataFrame(curr_data, columns=['v1', 'v2', 'v3'])
    next_centroids = init_centroids(pd_DataFrame_data, 2)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not next_centroids.equals(old_centroids):
        old_centroids = next_centroids

        labels = get_labels(pd_DataFrame_data, next_centroids)
        next_centroids = new_centroids(pd_DataFrame_data, labels, 2)

    centroids.append(next_centroids.iloc[:, 0].values.tolist())
    centroids.append(next_centroids.iloc[:, 1].values.tolist())
    return centroids


# use bisecting kmeans method to generate clusters
def bisecting_kmeans(curr_data, k):
    # initialize cluster list using initial centroids
    # data points in cluster, centroid, SSE
    clusters = [[curr_data, 0, 0]]

    # iterative cluster generation
    for i in range(0, k - 1):
        # cluster with highest SSE will be last
        # split the last cluster in the list
        split_centroid = clusters.pop()[0]

        # create temporary lists for each new centroid
        new_centroid1 = []
        new_centroid2 = []

        # find 2 new centroids
        centroids = simple_KMeans(split_centroid)

        # assign data points to each centroid
        labels = assign_centroid(split_centroid, centroids)

        # add data to each new cluster based on cluster labels
        for j in range(0, len(labels)):
            if labels[j] == 0:
                new_centroid1.append(split_centroid[j])
            else:
                new_centroid2.append(split_centroid[j])

        # calculate SSE for each new centroid
        new_centroid1_SSE = calc_SSE(new_centroid1, centroids[0])
        new_centroid2_SSE = calc_SSE(new_centroid2, centroids[1])

        # add the new clusters to the list
        clusters.append([new_centroid1, centroids[0], new_centroid1_SSE])
        clusters.append([new_centroid2, centroids[1], new_centroid2_SSE])

        # sort the list so highest SSE is last
        clusters.sort(key=lambda e: e[2])

    # build the full list of centroids
    # format for centroid output is [data points] [centroid coordinates] [centroid SSE] [centroid ID]
    all_centroids = []
    for a in range(0, len(clusters)):
        all_centroids.append(clusters[a])
        all_centroids[a].append(a)

    return all_centroids


def visualization():
    return 'Teddy, insert here'


def main():
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

    # convert data to list for ease of usage
    list_data = clean_data.values.tolist()

    #  TEMPORARY ASSIGNMENT FOR TESTING clusters from silhouette approach
    k = silhouette_meth(filename)
    # ('k: ' + str(k))
    # print()

    # format for centroid output is [data points] [centroid coordinates] [centroid SSE] [centroid ID]
    centroids = bisecting_kmeans(list_data, k)

    # display each centroid
    # for c in range(0, len(centroids)):
    # (centroids[c])
    # print()

    visualization()


if __name__ == '__main__':
    main()
