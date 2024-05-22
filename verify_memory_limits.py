from dask_cuda import LocalCUDACluster
from dask.distributed import Client, get_worker, get_client

def get_memory_info():
    worker = get_worker()
    return worker.nthreads, worker.memory_limit

def run_on_workers():
    client = get_client()
    return client.run(get_memory_info)

def check_memory():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    result = client.run(run_on_workers)
    print(result)

if __name__ == '__main__':
    check_memory()
