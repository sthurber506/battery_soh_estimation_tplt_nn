import dask.array as da
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, performance_report
from cuml.cluster import KMeans
import time

if __name__ == "__main__":
    # Initialize Dask cluster with LocalCUDACluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Connected to Dask cluster")

    # Increase task complexity to require more resources
    n_samples = 30_000_000  # Adjusted to ensure the task requires multiple machines
    n_features = 200
    n_clusters = 50

    # Generate random data
    X = np.random.rand(n_samples, n_features).astype(np.float32)

    # Distribute data using Dask
    dx = da.from_array(X, chunks=(n_samples // 30, n_features))

    # K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init="scalable-k-means++", random_state=0)

    # Measure execution time
    with performance_report(filename="dask_benchmark_report.html"):
        start_time = time.time()
        kmeans.fit(dx)
        end_time = time.time()

    print(f"K-means clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Cluster centers: {kmeans.cluster_centers_}")
