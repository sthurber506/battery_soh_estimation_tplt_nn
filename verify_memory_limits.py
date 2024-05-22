from dask.distributed import Client

def get_memory_info():
    from distributed import get_worker
    try:
        worker = get_worker()
        return f"Worker Address: {worker.address}, Threads: {worker.nthreads}, Memory Limit: {worker.memory_manager.memory_limit}"
    except ValueError:
        return "No worker found"

def check_memory():
    client = Client('tcp://localhost:8786')  # Update with your scheduler address if different
    print("Scheduler Address: tcp://localhost:8786")

    # Run the `get_memory_info` function on all workers
    result = client.run(get_memory_info)
    
    for worker, info in result.items():
        print(f"{worker}: {info}")

if __name__ == '__main__':
    check_memory()
