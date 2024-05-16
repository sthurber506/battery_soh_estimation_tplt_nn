import logging
import dask
from dask.distributed import Client
import dask.array as da

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log_worker_status(client):
    workers = client.scheduler_info()['workers']
    for worker, info in workers.items():
        logger.info(f"Worker {worker} has {info['nthreads']} threads and {info['memory_limit'] / 1e9:.2f} GB memory")

# Connect to the cluster
scheduler_address = "tcp://100.82.76.42:8786"
client = Client(scheduler_address, timeout=60)

# Log worker status
log_worker_status(client)

# Create a larger Dask array for testing
n_samples = 10000000
n_features = 200
X = da.random.random((n_samples, n_features), chunks=(n_samples // len(client.scheduler_info()['workers']), n_features))
X = X.persist()
client.wait_for_workers(1)

logger.info("Created Dask array")

# Perform a more intensive computation
try:
    X_sum = X.sum(axis=0).compute()
    logger.info(f"Sum: {X_sum}")
except Exception as e:
    logger.error(f"Computation failed: {e}")

# Log worker status after computation
log_worker_status(client)
