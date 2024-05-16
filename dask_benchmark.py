import dask.array as da
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, performance_report, get_task_stream, wait
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

    # Create and distribute data using Dask
    dx = da.random.random((n_samples, n_features), chunks=(n_samples // 100, n_features))
    dx = dx.persist()  # Persist the data in the cluster memory
    wait(dx)  # Ensure all data is distributed and ready

    # Log the status of the distributed array
    logging.info(f"Distributed array with shape {dx.shape} and chunks {dx.chunks}")

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
