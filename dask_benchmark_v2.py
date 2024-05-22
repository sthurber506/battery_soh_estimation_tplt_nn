from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import logging
import numpy as np
import dask.array as da
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log_worker_status(client):
    workers = client.scheduler_info()['workers']
    for worker, info in workers.items():
        logger.info(f"Worker {worker} has {info['nthreads']} threads and {info['memory_limit'] / 1e9:.2f} GB memory")

# Connect to the cluster
client = Client("tcp://localhost:8786")

# Log worker status
log_worker_status(client)

n_samples = 1000000
n_features = 200
chunk_size = n_samples // (len(client.scheduler_info()['workers']) * 2)

# Create a large Dask array with optimized chunk size
X = da.random.random((n_samples, n_features), chunks=(chunk_size, n_features))
X = X.persist()
wait(X)

logger.info("Created Dask array")

# Scatter the data to all workers
futures = client.scatter(X, broadcast=True)
wait(futures)

# Perform a computation
start_time = time.time()
X_mean = X.mean().compute()
end_time = time.time()

logger.info(f"Mean: {X_mean}")
logger.info(f"Computation took {end_time - start_time:.2f} seconds")
