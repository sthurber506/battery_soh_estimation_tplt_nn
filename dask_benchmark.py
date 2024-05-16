from dask.distributed import Client
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Connect to the cluster
client = Client("tcp://172.29.184.162:8786", timeout=60)  # Increase timeout to 60 seconds

# Log worker status
def log_worker_status(client):
    workers = client.scheduler_info()['workers']
    for worker, info in workers.items():
        logger.info(f"Worker {worker} has {info['nthreads']} threads and {info['memory_limit'] / 1e9:.2f} GB memory")

log_worker_status(client)
