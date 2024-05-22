from dask_cuda import LocalCUDACluster
from dask.distributed import Client
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

# Create large Dask arrays with smaller chunks
chunks = (n_samples // (len(client.scheduler_info()['workers']) * 4), n_features)
X = da.random.random((n_samples, n_features), chunks=chunks)
Y = da.random.random((n_features, n_samples), chunks=chunks)

X = X.persist()
Y = Y.persist()
client.wait_for_workers(1)

logger.info("Created Dask arrays")

# Perform a computation
start_time = time.time()
X_mean = X.mean().compute()
X_sum = X.sum(axis=0).compute()
X_dot_Y = da.dot(X, Y).compute()
end_time = time.time()

logger.info(f"Mean: {X_mean}")
logger.info(f"Sum: {X_sum}")
logger.info(f"Dot product shape: {X_dot_Y.shape}")
logger.info(f"Computation took {end_time - start_time:.2f} seconds")
