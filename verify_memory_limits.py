from dask_cuda import LocalCUDACluster
from dask.distributed import Client

def check_memory():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    result = client.run(lambda: (client.nthreads(), client.memory_limit))
    print(result)

if __name__ == '__main__':
    check_memory()
