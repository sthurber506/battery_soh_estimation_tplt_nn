from distributed import get_worker

def get_memory_info():
    try:
        worker = get_worker()
        return f"Worker Address: {worker.address}, Threads: {worker.nthreads}, Memory Limit: {worker.memory_manager.memory_limit}"
    except ValueError:
        return "No worker found"

# This script should not be executed directly; it's meant to be called via `client.run()`
