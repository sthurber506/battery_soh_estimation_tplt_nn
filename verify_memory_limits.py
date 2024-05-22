from dask.distributed import Client

def check_memory():
    client = Client('tcp://100.82.76.42:8786')  # Update with your scheduler address if different
    print("Scheduler Address: tcp://100.82.76.42:8786")

    # Get the worker information from the scheduler
    info = client.scheduler_info()
    workers = info['workers']

    for worker, details in workers.items():
        print(f"Worker Address: {worker}")
        print(f"  Threads: {details['nthreads']}")
        print(f"  Memory Limit: {details['memory_limit'] / 1e9} GB")
        print(f"  Memory: {details['memory'] / 1e9} GB")

if __name__ == '__main__':
    check_memory()
