from dask.distributed import Client

def get_memory_info():
    import os
    from distributed import get_worker
    worker = get_worker()
    return {
        'worker_address': worker.address,
        'nthreads': worker.nthreads,
        'memory_limit': worker.memory_manager.memory_limit,
        'memory_info': os.popen('free -h').read()
    }

def check_memory():
    client = Client('tcp://localhost:8786')  # Update with your scheduler address if different
    result = client.run(get_memory_info)
    for worker, info in result.items():
        print(f"Worker Address: {info['worker_address']}, Threads: {info['nthreads']}, Memory Limit: {info['memory_limit']}")
        print(f"Memory Info: {info['memory_info']}")

if __name__ == '__main__':
    check_memory()
