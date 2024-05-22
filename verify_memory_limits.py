from dask.distributed import Client, get_worker

def get_memory_info():
    worker = get_worker()
    return {
        'nthreads': worker.nthreads,
        'memory_limit': worker.memory_limit,
    }

def check_memory():
    client = Client('tcp://localhost:8786')  # Update with your scheduler address if different
    result = client.run(get_memory_info)
    for worker, info in result.items():
        print(f"Worker: {worker}, Threads: {info['nthreads']}, Memory Limit: {info['memory_limit']}")

if __name__ == '__main__':
    check_memory()
