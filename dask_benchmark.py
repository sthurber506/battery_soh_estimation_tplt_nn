import dask.array as da
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, performance_report, get_task_stream
from cuml.cluster import KMeans
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def log_worker_status(client):
    # Get the information about the cluster
    workers = client.scheduler_info()["workers"]
    for worker, info in workers.items():
        logging.info(f"Worker {worker} has {info['nthreads']} threads and {info['memory_limit'] / 1e9:.2f} GB memory")

    # Print the task statuses
    status = client.scheduler_info()["status"]
    logging.info(f"Cluster status: {status}")

if __name__ == "__main__":
    # Initialize Dask cluster with LocalCUDACluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    logging.info("Connected to Dask cluster")

    # Log the initial state of the cluster
    log_worker_status(client)

    # Increase task complexity
    n_samples = 50_000_000  # More samples
    n_features = 200
    n_clusters = 100

    # Generate random data
    X = np.random.rand(n_samples, n_features).astype(np.float32)

    # Distribute data using Dask
    dx = da.from_array(X, chunks=(n_samples // 100, n_features))

    # K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, init="scalable-k-means++", random_state=0)

    # Measure execution time and monitor task distribution
    with performance_report(filename="dask_benchmark_report.html"):
        with get_task_stream(plot="dask_benchmark_task_stream.html") as ts:
            start_time = time.time()
            kmeans.fit(dx)
            end_time = time.time()

    logging.info(f"K-means clustering completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Cluster centers: {kmeans.cluster_centers_}")

    # Log the final state of the cluster
    log_worker_status(client)

    # Print task stream results
    for task in ts.data:
        logging.info(f"Task {task['key']} executed on {task['worker']} in {task['end'] - task['start']:.2f} seconds")
