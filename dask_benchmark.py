import dask.array as da
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, performance_report
from cuml.cluster import KMeans
import time

if __name__ == "__main__":
    # Initialize Dask cluster with LocalCUDACluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Connected to Dask cluster")

    # Increase task complexity
    n_samples = 30_000_000  # Number of samples
    n_features = 200
    n_clusters = 10

    # Create a Dask array directly with smaller chunks
    dx = da.random.random((n_samples, n_features), chunks=(n_samples // 3000, n_features)).astype('float32')

    # Persist the Dask array to ensure intermediate results are stored
    dx = dx.persist()

    # K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init="scalable-k-means++", random_state=0)

    # Measure execution time
    start_time = time.time()
    with performance_report(filename="dask_benchmark_report.html"):
        kmeans.fit(dx)
    end_time = time.time()

    print(f"K-means clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Cluster centers: {kmeans.cluster_centers_}")
