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

def log_memory_usage(client):
    workers = client.scheduler_info()['workers']
    for worker, info in workers.items():
        if 'memory' in info:
            managed_memory = info['memory']['managed_in_memory'] / 1e9
            unmanaged_memory = info['memory']['unmanaged'] / 1e9
            logger.info(f"Worker {worker} managed memory: {managed_memory:.2f} GB, unmanaged memory: {unmanaged_memory:.2f} GB")
        else:
            logger.warning(f"Worker {worker} does not have memory information available.")

# Connect to the cluster
client = Client("tcp://172.29.184.162:8786", timeout=60)  # Increase timeout to 60 seconds

# Log worker status
log_worker_status(client)

n_samples = 1000000  # Smaller size to avoid memory issues
n_features = 200

# Create a Dask array with appropriate chunk size
chunks_per_worker = max(n_samples // (len(client.scheduler_info()['workers']) * 10), 1)
X = da.random.random((n_samples, n_features), chunks=(chunks_per_worker, n_features))
X = X.persist()
client.wait_for_workers(1)

logger.info("Created Dask array")

# Log memory usage before computation
log_memory_usage(client)

# Perform a computation
start_time = time.time()
try:
    X_mean = X.mean().compute()
    logger.info(f"Mean: {X_mean}")
except Exception as e:
    logger.error(f"Computation failed: {e}")
end_time = time.time()

logger.info(f"Computation took {end_time - start_time:.2f} seconds")

# Log worker status and memory usage after computation
log_worker_status(client)
log_memory_usage(client)
