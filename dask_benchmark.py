import dask
import time
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.distributed import performance_report
from cuml.cluster import KMeans

if __name__ == "__main__":
    # Initialize Dask cluster with LocalCUDACluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Connected to Dask cluster")

    # Generate random data
    n_samples = 1_000_000
    n_features = 100
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    
    # Distribute data using Dask
    dX = dask.array.from_array(X, chunks=(n_samples // 10, n_features))
    
    # K-means clustering
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, init="scalable-k-means++", random_state=0)

    with performance_report(filename="dask_benchmark_report.html"):
        start_time = time.time()
        kmeans.fit(dX)
        end_time = time.time()
        print(f"K-means clustering completed in {end_time - start_time:.2f} seconds")
        print(f"Cluster centers: {kmeans.cluster_centers_}")
