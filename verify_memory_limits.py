from dask_cuda import LocalCUDACluster
from dask.distributed import Client

def get_memory_info():
    from distributed import get_worker
    try:
        worker = get_worker()
        return worker.nthreads, worker.memory_limit
    except ValueError:
        return "No worker found"

def check_memory():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    result = client.run_on_scheduler(lambda: client.run(get_memory_info))
    print(result)

if __name__ == '__main__':
    check_memory()
