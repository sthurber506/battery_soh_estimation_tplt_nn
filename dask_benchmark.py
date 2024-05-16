import time
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, performance_report
import dask_cudf
import cuml

# Start measuring time
start_time = time.time()

# Connect to the Dask cluster
cluster = LocalCUDACluster()
client = Client(cluster)
print("Connected to Dask cluster")

# Generate some synthetic data
n_samples = 1_000_000
n_features = 50
X, _ = cuml.datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=8, cluster_std=0.1, random_state=0)
df = dask_cudf.from_cudf(X, npartitions=8)

# Generate a performance report
with performance_report(filename="dask_report.html"):
    # Perform K-means clustering
    kmeans = cuml.dask.cluster.KMeans(n_clusters=8)
    kmeans.fit(df)
    print("K-means clustering completed")
    print("Cluster centers:", kmeans.cluster_centers_)

# End measuring time
end_time = time.time()

# Print the total execution time
print(f"Total execution time: {end_time - start_time} seconds")
