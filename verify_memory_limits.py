# verify_memory_limits_worker.py

from distributed import get_worker
import os

def get_memory_info():
    worker = get_worker()
    print(f"Worker Address: {worker.address}")
    print(f"Threads: {worker.nthreads}")
    print(f"Memory Limit: {worker.memory_manager.memory_limit}")
    print("Memory Info: ")
    print(os.popen('free -h').read())

if __name__ == '__main__':
    get_memory_info()
