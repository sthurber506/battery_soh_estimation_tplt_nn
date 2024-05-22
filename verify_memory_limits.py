from dask.distributed import Client

def get_memory_info():
    worker = get_worker()
    return worker.nthreads, worker.memory_limit

def check_memory():
    client = Client('tcp://localhost:8786')  # Update with your scheduler address if different
    result = client.run(get_memory_info)
    print(result)

if __name__ == '__main__':
    check_memory()
