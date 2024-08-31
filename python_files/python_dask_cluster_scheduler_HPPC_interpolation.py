import dask
import dask_cudf
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client
from dask.distributed import wait
from dask.distributed import progress
from dask.diagnostics import ProgressBar

import os
import subprocess
from pathlib import Path
import pprint
import scipy.io
import cupy as cp
import cudf
import numpy as np
import tables
import pickle
import json
import gzip
import re
import gc

import datetime
import dateutil
import time

import math

import pandas as pd
# Set the maximum column width to a large number
pd.set_option('display.max_colwidth', 1000)
# Set the maximum rows to a large number
pd.set_option('display.max_rows', 1000)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Function to confirm Dask settings
def confirm_dask_settings():
    # Refresh the configuration to ensure changes are applied
    dask.config.refresh()
    dask.config.set({"config_file": "/mnt/shared/dask_config.yaml"})

    # Connect to the Dask scheduler
    client = Client("tcp://100.82.123.17:8786")
    client.restart()  # Restart the client to clear any previous state
    
    # Verify the configuration
    memory_target = client.run_on_scheduler(lambda: dask.config.get("distributed.worker.memory.target"))
    work_stealing = client.run_on_scheduler(lambda: dask.config.get("distributed.scheduler.work-stealing"))
    dataframe_backend = dask.config.get("dataframe.backend")
    array_backend = dask.config.get("array.backend")

    # Explanation
    # client.run_on_scheduler(func): This method runs the function func on the scheduler and returns the result. It is useful for getting configuration or status information that is only available on the scheduler.
    # lambda: dask.config.get("distributed.worker.memory.target"): The lambda here is a small anonymous function that takes no arguments and returns the value of the configuration setting distributed.worker.memory.target.
    # lambda: dask.config.get("distributed.scheduler.work-stealing"): Similarly, this lambda returns the value of the configuration setting distributed.scheduler.work-stealing.
    # Why Use a Lambda Function?
    # Anonymous Function: The lambda function is an anonymous function, meaning it does not have a name. It is useful for short functions that are used only once and do not need to be named.
    # Inline Definition: Using a lambda allows defining the function inline where it is used, making the code more concise.
    # Delayed Execution: The lambda function is executed when run_on_scheduler is called. This ensures that the configuration is fetched directly from the scheduler at the time of execution.

    # Print the configuration to verify
    print(f"Memory target: {memory_target}")
    print(f"Work stealing: {work_stealing}")
    print(f"Dataframe backend: {dataframe_backend}")
    print(f"Array backend: {array_backend}\n")  # Insert new line here

# Function to set path variables
def set_path_variables():
    path_nfs_root = '/mnt/shared'

    path_simulink_data_gen = os.path.join(path_nfs_root, 'simulink_data_generator')

    path_base_battery_model_params = os.path.join(path_simulink_data_gen, 'base_battery_model_params')
    path_HPPC_base_params = os.path.join(path_simulink_data_gen, 'base_battery_model_params_HPPC')
    path_HPPC_interpolated_params = os.path.join(path_simulink_data_gen, 'HPPC_interpolated_params')
    path_matlab_scripts = os.path.join(path_simulink_data_gen, 'matlab_scripts')  

    path_temp_slices = os.path.join(path_HPPC_interpolated_params, 'temp_slices')

    # Create the directories if they do not exist, but if they do exist, do not overwrite them
    os.makedirs(path_simulink_data_gen, exist_ok=True)
    os.makedirs(path_base_battery_model_params, exist_ok=True)
    os.makedirs(path_HPPC_base_params, exist_ok=True)
    os.makedirs(path_HPPC_interpolated_params, exist_ok=True)
    os.makedirs(path_temp_slices, exist_ok=True)
    os.makedirs(path_matlab_scripts, exist_ok=True)

    # Return a dictionary of the paths
    return {
        "nfs_root": path_nfs_root,
        "simulink_data_gen": path_simulink_data_gen,
        "base_battery_model_params": path_base_battery_model_params,
        "HPPC_base_params": path_HPPC_base_params,
        "HPPC_interpolated_params": path_HPPC_interpolated_params,
        "temp_slices": path_temp_slices,
        "matlab_scripts": path_matlab_scripts
    }


# Confirm Dask settings
confirm_dask_settings()

# Set path variables
paths = set_path_variables()
































# Define worker resource capabilities with additional GPU parameters
worker_capabilities = {
    "100.82.123.17": {
        "GPU_VRAM": 8,                  # Total GPU memory in GB
        "CUDA_Cores": 5888,                # CUDA cores
        "Tensor_Cores": 184,                # Tensor cores
        "GPU_Clock_Base": 1500,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1725,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 1750,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 448,           # Memory bandwidth in GB/s
        "CPU_Cores": 16,                 # Number of CPU cores
        "CPU_Threads": 24,                 # Number of CPU threads
        "RAM": 48,                         # System RAM in GB
        "BaseClock": 2.40,                 # CPU base clock speed in GHz
        "BoostClock": 5.10,                # CPU boost clock speed in GHz
        "Disk_IOPS": 124,               # Disk IOPS
        "Disk_Bandwidth": 483,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 10.46,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 13.93,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 1.4,       # Memory in MB
        "L2_Cache": 26,       # Memory in MB
        "L3_Cache": 30       # Memory in MB
    },
    "100.73.226.71": {
        "GPU_VRAM": 16,                  # Total GPU memory in GB
        "CUDA_Cores": 6144,                # CUDA cores
        "Tensor_Cores": 192,                # Tensor cores
        "GPU_Clock_Base": 735,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1560,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 1750,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 448,           # Memory bandwidth in GB/s
        "CPU_Cores": 10,                 # Number of CPU cores
        "CPU_Threads": 20,                 # Number of CPU threads
        "RAM": 48,                         # System RAM in GB
        "BaseClock": 2.40,                 # CPU base clock speed in GHz
        "BoostClock": 3.20,                # CPU boost clock speed in GHz
        "Disk_IOPS": 77.5,               # Disk IOPS
        "Disk_Bandwidth": 303,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 4.36,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 5.63,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.64,       # Memory in MB
        "L2_Cache": 10.24,       # Memory in MB
        "L3_Cache": 14       # Memory in MB
    },
    "100.65.124.63": {
        "GPU_VRAM": 8,                  # Total GPU memory in GB
        "CUDA_Cores": 896,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 1065,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1395,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 1250,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 160,           # Memory bandwidth in GB/s
        "CPU_Cores": 4,                 # Number of CPU cores
        "CPU_Threads": 8,                 # Number of CPU threads
        "RAM": 24,                         # System RAM in GB
        "BaseClock": 4.10,                 # CPU base clock speed in GHz
        "BoostClock": 4.60,                # CPU boost clock speed in GHz
        "Disk_IOPS": 86.8,               # Disk IOPS
        "Disk_Bandwidth": 339,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 5.22,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 8.72,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.256,       # Memory in MB
        "L2_Cache": 4.096,       # Memory in MB
        "L3_Cache": 8       # Memory in MB
    },
    "100.72.43.78": {
        "GPU_VRAM": 8,                  # Total GPU memory in GB
        "CUDA_Cores": 896,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 1065,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1395,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 1250,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 160,           # Memory bandwidth in GB/s
        "CPU_Cores": 4,                 # Number of CPU cores
        "CPU_Threads": 8,                 # Number of CPU threads
        "RAM": 24,                         # System RAM in GB
        "BaseClock": 4.10,                 # CPU base clock speed in GHz
        "BoostClock": 4.60,                # CPU boost clock speed in GHz
        "Disk_IOPS": 81.9,               # Disk IOPS
        "Disk_Bandwidth": 320,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 6.48,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 9.87,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.256,       # Memory in MB
        "L2_Cache": 4.096,       # Memory in MB
        "L3_Cache": 8       # Memory in MB
    },
    "100.113.204.47": {
        "GPU_VRAM": 8,                  # Total GPU memory in GB
        "CUDA_Cores": 5120,                # CUDA cores
        "Tensor_Cores": 160,                # Tensor cores
        "GPU_Clock_Base": 1110,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1560,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 1750,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 448,           # Memory bandwidth in GB/s
        "CPU_Cores": 8,                 # Number of CPU cores
        "CPU_Threads": 16,                 # Number of CPU threads
        "RAM": 15,                         # System RAM in GB
        "BaseClock": 3.30,                 # CPU base clock speed in GHz
        "BoostClock": 4.60,                # CPU boost clock speed in GHz
        "Disk_IOPS": 90.8,               # Disk IOPS
        "Disk_Bandwidth": 355,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 5.43,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 6.47,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.512,       # Memory in MB
        "L2_Cache": 4.096,       # Memory in MB
        "L3_Cache": 16       # Memory in MB
    },
    "100.75.156.93": {
        "GPU_VRAM": 6,                  # Total GPU memory in GB
        "CUDA_Cores": 1408,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 1530,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 1785,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 2001,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 192.1,           # Memory bandwidth in GB/s
        "CPU_Cores": 4,                 # Number of CPU cores
        "CPU_Threads": 8,                 # Number of CPU threads
        "RAM": 9,                         # System RAM in GB
        "BaseClock": 4.00,                 # CPU base clock speed in GHz
        "BoostClock": 4.20,                # CPU boost clock speed in GHz
        "Disk_IOPS": 50.7,               # Disk IOPS
        "Disk_Bandwidth": 198,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 7.41,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 8.54,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.256,       # Memory in MB
        "L2_Cache": 1.024,       # Memory in MB
        "L3_Cache": 8       # Memory in MB
    },
    "100.82.151.63": {
        "GPU_VRAM": 0,                  # Total GPU memory in GB
        "CUDA_Cores": 0,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 0,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 0,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 0,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 0,           # Memory bandwidth in GB/s
        "CPU_Cores": 4,                 # Number of CPU cores
        "CPU_Threads": 8,                 # Number of CPU threads
        "RAM": 9,                         # System RAM in GB
        "BaseClock": 3.6,                 # CPU base clock speed in GHz
        "BoostClock": 4.2,                # CPU boost clock speed in GHz
        "Disk_IOPS": 60,               # Disk IOPS
        "Disk_Bandwidth": 234,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 0,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 0,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.256,       # Memory in MB
        "L2_Cache": 1.024,       # Memory in MB
        "L3_Cache": 8       # Memory in MB
    },
    "100.120.1.68": {
        "GPU_VRAM": 0,                  # Total GPU memory in GB
        "CUDA_Cores": 0,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 0,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 0,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 0,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 0,           # Memory bandwidth in GB/s
        "CPU_Cores": 6,                 # Number of CPU cores
        "CPU_Threads": 12,                 # Number of CPU threads
        "RAM": 9,                         # System RAM in GB
        "BaseClock": 2.60,                 # CPU base clock speed in GHz
        "BoostClock": 4.40,                # CPU boost clock speed in GHz
        "Disk_IOPS": 73.6,               # Disk IOPS
        "Disk_Bandwidth": 287,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 0,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 0,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.480,       # Memory in MB
        "L2_Cache": 3.072,       # Memory in MB
        "L3_Cache": 12       # Memory in MB
    },
    "100.123.66.46": {
        "GPU_VRAM": 0,                  # Total GPU memory in GB
        "CUDA_Cores": 0,                # CUDA cores
        "Tensor_Cores": 0,                # Tensor cores
        "GPU_Clock_Base": 0,                 # GPU clock speed in MHz
        "GPU_Clock_Boost": 0,                 # GPU clock speed in MHz
        "GPU_Memory_Clock": 0,                 # GPU clock speed in MHz
        "Memory_Bandwidth": 0,           # Memory bandwidth in GB/s
        "CPU_Cores": 4,                 # Number of CPU cores
        "CPU_Threads": 4,                 # Number of CPU threads
        "RAM": 9,                         # System RAM in GB
        "BaseClock": 3.40,                 # CPU base clock speed in GHz
        "BoostClock": 4.00,                # CPU boost clock speed in GHz
        "Disk_IOPS": 35.2,               # Disk IOPS
        "Disk_Bandwidth": 137,             # Disk bandwidth in MB/s
        "CPU_to_GPU_Bandwidth": 0,      # CPU to GPU bandwidth in GB/s
        "GPU_to_CPU_Bandwidth": 0,       # GPU to CPU bandwidth in GB/s
        "L1_Cache": 0.256,       # Memory in MB
        "L2_Cache": 1.024,       # Memory in MB
        "L3_Cache": 8       # Memory in MB
    }
}



























# Define the MATLAB executable path
matlab_executable = "/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe"

# Define the MATLAB script paths
matlab_scripts_path = "/mnt/shared/simulink_data_generator/matlab_scripts"
interpolation_function_name = "python_dask_run_HPPC_interpolation_iterations"
collection_function_name = "python_dask_collect_HPPC_results"

# Define Dask client connection
client = Client("tcp://100.82.123.17:8786")

# Function to run MATLAB command on each worker
def run_matlab_command(worker_id, start_idx, end_idx, profile_name, training_set_ratio):
    command = f'"{matlab_executable}" -batch "addpath(\'{matlab_scripts_path}\'); {interpolation_function_name}({start_idx},{end_idx},\'{profile_name}\',{training_set_ratio})"'
    print(f"Submitting command for worker {worker_id}: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"Worker {worker_id} completed successfully.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Worker {worker_id} failed with error.")
        print(f"Error output: {e.stderr}")
        raise e

# Set the total number of iterations to perform
total_iterations = 1000
training_set_ratio = 0.8  # Example ratio for training set
profile_name = 'LocalProfile1'  # Example profile name for the MATLAB parallel pool

# Function to calculate chunk sizes
def calculate_chunk_sizes(worker_capabilities, total_iterations):
    total_cores = sum(worker["CPU_Cores"] for worker in worker_capabilities.values())
    
    chunk_sizes = {}
    start_idx = 0

    for worker_id, capabilities in worker_capabilities.items():
        worker_cores = capabilities["CPU_Cores"]
        
        # Calculate the proportion of total iterations this worker should handle
        proportion = worker_cores / total_cores
        
        # Calculate the base chunk size for this worker
        chunk_size = math.ceil(total_iterations * proportion)
        
        # Ensure the chunk size is a multiple of the number of cores
        chunk_size = max(1, round(chunk_size / worker_cores) * worker_cores)
        
        end_idx = start_idx + chunk_size - 1
        chunk_sizes[worker_id] = (start_idx, end_idx)
        start_idx = end_idx + 1

    return chunk_sizes

# Calculate the chunk sizes
chunk_sizes = calculate_chunk_sizes(worker_capabilities, total_iterations)
pprint.pprint(chunk_sizes)

# Distribute tasks
futures = []
for worker_id, (start_idx, end_idx) in chunk_sizes.items():
    futures.append(client.submit(
        run_matlab_command,
        worker_id,
        start_idx,
        end_idx,
        profile_name,
        training_set_ratio,
        workers=[worker_id]
    ))

# Wait for the tasks to complete
progress(futures)
results = client.gather(futures)

# Print results
for worker, result in zip(chunk_sizes.keys(), results):
    print(f"Result from worker {worker}: {result}")

temp_slices_path = "/mnt/shared/simulink_data_generator/HPPC_interpolated_params/temp_slices"
output_files = os.listdir(temp_slices_path)

print("Generated Output Files:")
for file in output_files:
    print(file)

def collect_results():
    command = f'"{matlab_executable}" -batch "addpath(\'{matlab_scripts_path}\'); {collection_function_name}"'
    print(f"Submitting collection command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Collection script completed successfully.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Collection script failed with error.")
        print(f"Error output: {e.stderr}")
        raise e

# Run collection script
collect_results()





































# Run save interpolated parameters script
def save_interpolated_parameters():
    command = f'"{matlab_executable}" -batch "addpath(\'{matlab_scripts_path}\'); python_dask_save_interpolated_parameters"'
    print(f"Submitting save parameters command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Save parameters script completed successfully.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Save parameters script failed with error.")
        print(f"Error output: {e.stderr}")
        raise e

# Run save interpolated parameters script
save_interpolated_parameters()


