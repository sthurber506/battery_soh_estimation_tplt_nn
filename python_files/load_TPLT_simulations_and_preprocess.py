import dask
import dask_cudf
import dask.dataframe as dd
import dask.bag as db
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client
from dask.distributed import wait
from dask.distributed import progress
from dask.distributed import get_worker
from dask.diagnostics import ProgressBar

import os
import subprocess
import shutil
from pathlib import Path
import pprint

import scipy.io
from scipy.io import loadmat

import cupy as cp
import cudf
import numpy as np
import tables
import pickle
import json
import gzip
import re
import gc
import itertools

from IPython.display import display

import datetime
import dateutil
import time

import math

import h5py

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
    path_simulink_model = os.path.join(path_simulink_data_gen, 'simulink_model')
    path_simulink_results = os.path.join(path_simulink_data_gen, 'simulink_results')

    path_temp_slices = os.path.join(path_HPPC_interpolated_params, 'temp_slices')

    path_combined_data = os.path.join(path_simulink_results, 'combined_data')
    path_combined_ddf = os.path.join(path_simulink_results, 'combined_ddf')
    path_combined_ddf_pickle = os.path.join(path_simulink_results, 'combined_ddf_pickle')
    path_chunked_mat_files = os.path.join(path_simulink_results, 'chunked_mat_files')
    path_parquet_files = os.path.join(path_simulink_results, 'parquet_files')
    path_parquet_files_modified = os.path.join(path_simulink_results, 'parquet_files_modified')

    # Create the directories if they do not exist, but if they do exist, do not overwrite them
    os.makedirs(path_simulink_data_gen, exist_ok=True)
    os.makedirs(path_base_battery_model_params, exist_ok=True)
    os.makedirs(path_HPPC_base_params, exist_ok=True)
    os.makedirs(path_HPPC_interpolated_params, exist_ok=True)
    os.makedirs(path_temp_slices, exist_ok=True)
    os.makedirs(path_matlab_scripts, exist_ok=True)
    os.makedirs(path_simulink_model, exist_ok=True)
    os.makedirs(path_simulink_results, exist_ok=True)
    os.makedirs(path_combined_data, exist_ok=True)
    os.makedirs(path_combined_ddf, exist_ok=True)
    os.makedirs(path_combined_ddf_pickle, exist_ok=True)
    os.makedirs(path_chunked_mat_files, exist_ok=True)
    os.makedirs(path_parquet_files, exist_ok=True)
    os.makedirs(path_parquet_files_modified, exist_ok=True)

    # Return a dictionary of the paths
    return {
        "nfs_root": path_nfs_root,
        "simulink_data_gen": path_simulink_data_gen,
        "base_battery_model_params": path_base_battery_model_params,
        "HPPC_base_params": path_HPPC_base_params,
        "HPPC_interpolated_params": path_HPPC_interpolated_params,
        "temp_slices": path_temp_slices,
        "matlab_scripts": path_matlab_scripts,
        "simulink_model": path_simulink_model,
        "simulink_results": path_simulink_results,
        "combined_data": path_combined_data,
        "combined_ddf": path_combined_ddf,
        "combined_ddf_pickle": path_combined_ddf_pickle,
        "chunked_mat_files": path_chunked_mat_files,
        "parquet_files": path_parquet_files,
        "parquet_files_modified": path_parquet_files_modified
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


# Define Dask client connection
client = Client("tcp://100.82.123.17:8786")

















































# ONLY RUN THIS ONCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Directory containing .mat files in WSL from the paths dictionary under the key 'simulink_results'
mat_files_dir = paths['simulink_results']
parquet_output_dir = paths['parquet_files']

# Create the output folder "combined_data" in the same directory as the .mat files if it does not exist
output_dir = os.path.join(mat_files_dir, "combined_data")
os.makedirs(output_dir, exist_ok=True)

# Expected variables in .mat files
expected_variables = [
    'C_rate_initialized',
    'Capacity_initialized',
    'Current',
    'OCV',
    'SOC',
    'SOC_initialized',
    'SOC_rounded',
    'Temperature',
    'Time',
    'Voltage'
]

# Function to check if all expected variables are present in the .mat file
def check_file_integrity(file_path):
    try:
        mat_data = loadmat(file_path)
        if all(var in mat_data for var in expected_variables):
            return file_path  # Return the file path if it is valid
        else:
            os.remove(file_path)  # Delete the file if it does not contain all expected variables
            return None
    except:
        os.remove(file_path)  # Delete the file if an error occurs while loading
        return None

# Check all .mat files and collect valid ones using Dask
mat_files = [os.path.join(mat_files_dir, f) for f in os.listdir(mat_files_dir) if f.endswith('.mat')]
valid_files_futures = client.map(check_file_integrity, mat_files)
valid_files = client.gather(valid_files_futures)
valid_files = [f for f in valid_files if f is not None]

print(f"Found {len(mat_files)} .mat files")
print(f"Found {len(valid_files)} valid .mat files")



















































# Function to load a .mat file and convert it to a cuDF DataFrame
def load_mat_to_cudf(file_path, dataset_idx):
    mat_data = loadmat(file_path)
    data_dict = {}
    
    for key, value in mat_data.items():
        if not key.startswith('__'):
            data_dict[key] = value.flatten()  # Flatten the array to 1D
    
    df = cudf.DataFrame(data_dict)
    df['dataset_idx'] = dataset_idx  # Add a column to indicate the dataset index
    return df

# Parameters for batch processing
batch_size = 1000
num_batches = (len(valid_files) + batch_size - 1) // batch_size  # Ceiling division
# The above creates 71 batches of 1000 files each, with the last batch containing 200 files.
# Each of the resulting batch files are approximately 4.5 GB in size.
# The scheduler machine has an 8 GB GPU, so it can handle one batch at a time.
# So the idea is, after all batches are processed, we can load the entire dataset into memory and 
# perform operations on it quickly and then save it back to disk.

# Process files in batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(valid_files))
    batch_files = valid_files[start_idx:end_idx]

    # Load each batch of .mat files into cuDF DataFrames and convert to Dask-cuDF DataFrame
    dfs = [delayed(load_mat_to_cudf)(file_name, idx) for idx, file_name in enumerate(batch_files, start=start_idx)]
    batch_ddf = dask_cudf.from_delayed(dfs)

    # Monitor progress
    progress(batch_ddf)
    
    # Convert the Dask-cuDF DataFrame to a single cuDF DataFrame with progress bar
    with ProgressBar():
        batch_df = batch_ddf.compute()
    
    # Save the batch DataFrame as a single Parquet file
    parquet_file_path = os.path.join(parquet_output_dir, f'batch_{batch_idx + 1}.parquet')
    batch_df.to_parquet(parquet_file_path, index=False)
    
    print(f"Processed and saved batch {batch_idx + 1} out of {num_batches}")







































# Set a variable to store the nominal capacity of the battery
nominal_capacity = 2.84  # Ah

# Get the number of batch files in the parquet_files directory
num_batches = len([f for f in os.listdir(paths['parquet_files']) if f.endswith('.parquet')])
print(f"Number of batch files: {num_batches}")

# Create a for loop that loads each batch of the Parquet files and 
# converts them to a single cuDF DataFrame, and then adds new columns to the DataFrame.
# Finally, save the first 100,002 rows of the DataFrame to a CSV file.
for batch_idx in range(num_batches):
    # Define paths
    parquet_file_path = os.path.join(paths['parquet_files'], f'batch_{batch_idx + 1}.parquet')
    output_csv_path = os.path.join(paths['parquet_files_modified'], f'batch_{batch_idx + 1}_first_100002.csv')
    parquet_file_modified_path = os.path.join(paths['parquet_files_modified'], f'batch_{batch_idx + 1}.parquet')

    # Print a message to indicate the current batch being processed
    print(f"Processing batch {batch_idx + 1}...")

    # Invoke garbage collection to free up memory before loading the next batch
    gc.collect()
    
    # Load the Parquet file into a cuDF DataFrame
    df = cudf.read_parquet(parquet_file_path)

    # First, sort the DataFrame by 'dataset_idx' and 'Time' columns to ensure the data is ordered correctly
    df = df.sort_values(['dataset_idx', 'Time'])

    # The first new column to add is 'dVdt', which is the derivative of the 'Voltage' column with respect to 'Time'.
    # This can be calculated by taking the difference of the 'Voltage' column and dividing by the difference of the 'Time' column.
    df['dVdt'] = df['Voltage'].diff() / df['Time'].diff()

    # The next column to add is 'dt', which is the difference of the 'Time' column.
    df['dt'] = df['Time'].diff()

    # The next column to add will be titled 'tplt_V_max'. The caclulation for this column is as follows:
    # 1. Group the DataFrame by 'dataset_idx'
    # 2. For each group, filter the voltage values by the condition that both 'dVdt' and 'dt' are greater than 0
    # 3. Grab the maximum value from the filtered voltage values and assign it to the 'tplt_V_max' column for that group
    # 4. For the first row of the entire Dataframe, the 'dVdt' and 'dt' values are NaN, so the 'tplt_V_max' value will be NaN as well,
    #    so replace that NaN value with the 'tplt_V_max' value of the first group
    # 5. Ensure that the operations used to achieve the above steps are done for cuDF DataFrames, which cannot use 'apply' or 'lambda' functions
    
    # Filter rows where 'dVdt' and 'dt' are greater than 0. This step is necessary to apply the condition 
    # for calculating 'tplt_V_max' only on rows that meet the specified criteria, ensuring that 'tplt_V_max' is calculated 
    # from valid voltage values.
    filtered_df = df.query("dVdt > 0 and dt > 0")

    # After filtering, group the DataFrame by 'dataset_idx'. This creates a GroupBy object in cuDF, which allows for 
    # aggregation operations to be performed on each group separately, leveraging GPU acceleration.
    # For each group identified by a unique 'dataset_idx', we want to perform an aggregation operation.
    tplt_vmax_df = filtered_df.groupby('dataset_idx').agg({'Voltage': 'max'}).reset_index()
    # Here's a breakdown of the aggregation operation in the context of cuDF:
    # 1. `groupby('dataset_idx')`: This method groups the cuDF DataFrame based on unique values in the 'dataset_idx' column.
    #    Each group consists of all rows that have the same value in 'dataset_idx', similar to how it works in pandas but optimized for GPU.
    # 2. `.agg({'Voltage': 'max'})`: The `agg` method is used to apply one or more aggregation operations to the grouped data.
    #    - The argument `{'Voltage': 'max'}` is a dictionary where:
    #      - The key ('Voltage') specifies the column to aggregate.
    #      - The value ('max') specifies the aggregation function to apply, in this case, finding the maximum value.
    #    This tells cuDF to calculate the maximum 'Voltage' value for each group, utilizing GPU for the computation.
    # 3. `.reset_index()`: After aggregation, the result is a cuDF DataFrame where the index corresponds to the grouped keys ('dataset_idx').
    #    `reset_index()` is used to reset the index of this DataFrame to the default integer index, making 'dataset_idx' a column again
    #    and ensuring the aggregated DataFrame is in a suitable format for merging back into the original DataFrame.

    # Rename the 'Voltage' column in the aggregated DataFrame to 'tplt_V_max' to clearly indicate that this column 
    # represents the maximum voltage value for each group. This step makes the DataFrame ready for merging 
    # by aligning the column names.
    tplt_vmax_df = tplt_vmax_df.rename(columns={"Voltage": "tplt_V_max"})

    # Merge the vmax DataFrame back into the original DataFrame on 'dataset_idx'. This operation adds the 'tplt_V_max' 
    # column to the original DataFrame, aligning each 'tplt_V_max' value with its corresponding group. The merge is 
    # done with a 'left' join to ensure all rows from the original DataFrame are retained, even if they don't 
    # have a corresponding 'tplt_V_max' value.
    df = df.merge(tplt_vmax_df, on='dataset_idx', how='left')

    # Fill NaN values in the 'tplt_V_max' column. This step is necessary because the first row of the DataFrame and 
    # any groups that did not meet the filtering criteria will have NaN values for 'tplt_V_max'. Using 'ffill' (forward fill) 
    # propagates the last valid observation forward to next valid. This assumes that the 'tplt_V_max' value of the first 
    # group is representative for any initial NaN values, providing a reasonable estimate for 'tplt_V_max' where the 
    # direct calculation was not applicable.
    df['tplt_V_max'] = df['tplt_V_max'].ffill()

    # Re-sort the DataFrame after merging to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])

    # The next column to add will be titled 'tplt_V_min'. The calculation for this column is similar to 'tplt_V_max',
    # but the condition for filtering the rows is that both 'dVdt' and 'dt' are less than 0, and the aggregation
    # function is 'min' instead of 'max'.
    filtered_df = df.query("dVdt < 0 and dt > 0")
    tplt_vmin_df = filtered_df.groupby('dataset_idx').agg({'Voltage': 'min'}).reset_index()
    tplt_vmin_df = tplt_vmin_df.rename(columns={"Voltage": "tplt_V_min"})
    df = df.merge(tplt_vmin_df, on='dataset_idx', how='left')
    df['tplt_V_min'] = df['tplt_V_min'].ffill()

    # Re-sort the DataFrame after merging to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])

    # The next column to add will be titled 'tplt_delta_V_2', which is the difference between 'tplt_V_max' and 'tplt_V_min'.
    df['tplt_delta_V_2'] = df['tplt_V_max'] - df['tplt_V_min']

    # Re-sort the DataFrame after adding the new columns to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])

    # The C_rate_initialized column was created in the original MATLAB data generation script,
    # and it was calculated using the nominal Ah capacity value of the battery which is 2.84 Ah.
    # The TPLT paper calculates the C-rate by using delta_V_2 and some constants, and I assume this to be
    # an attempt to calculate the C-rate using the actual capacity of the battery instead of the nominal capacity.
    # Therefore, a new column titled 'C_rate_based_on_AHC_initialized' will be added to the DataFrame.
    # The calculation for AHC is as follows:
    # AHC = I/C_rate
    # where I is the current in Amperes and C_rate is the C-rate.
    # Therefore, C_rate = I/AHC, and the C_rate_based_on_AHC_initialized column will be calculated using this formula,
    # where I is the 'Current' column and AHC is the 'Capacity_initialized' column.
    # However, some precautions need to be taken to handle division by zero and NaN values.
    # The 'Current' column contains some zero values, which would result in division by zero, so these rows will be filtered out.
    # The two pulse load test discharges the battery, so the 'Current' column will be negative, so only negative values will be considered.
    # We should take the lowest value of 'Current' to calculate the C-rate.
    # The 'Capacity_initialized' column has no zero values.
    # If we filter out the zero values in the 'Current' column, we will have a different number of rows in the 'Current' and 'Capacity_initialized' columns.
    # Therefore, we will calculate the C-rate for each group of 'dataset_idx' values and merge the results back into the DataFrame.
    # For each dataset_idx group, we will calculate the C-rate using the minimum 'Current' value and the 'Capacity_initialized' value(which is constant for each group).
    # The 'C_rate_based_on_AHC_initialized' column will be added to the DataFrame with the calculated C-rate values.
    # The 'C_rate_based_on_AHC_initialized' value for each group should be calculated taking the absolute value of the minimum 'Current' value.
    # The result should be that for each dataset_idx group, the 'C_rate_based_on_AHC_initialized' value is the same for all rows in that group.
    filtered_df = df.query("Current < 0")
    c_rate_df = filtered_df.groupby('dataset_idx').agg({'Current': 'min'}).reset_index()
    c_rate_df = c_rate_df.rename(columns={"Current": "min_Current"})
    df = df.merge(c_rate_df, on='dataset_idx', how='left')
    df['min_Current'] = df['min_Current'].ffill()

    # Re-sort the DataFrame after merging to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])
    
    # Calculate the C-rate based on the minimum current and the initialized capacity
    df['C_rate_based_on_AHC_initialized'] = df['min_Current'].abs() / df['Capacity_initialized']

    # Re-sort the DataFrame after merging to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])

    # The next column to add will be titled 'SOH', which is the State of Health of the battery.
    # The calculation for SOH, not as a percentage, is as follows:
    # SOH = AHC_aged / AHC_nominal
    # where AHC_aged is the aged capacity of the battery and AHC_nominal is the nominal capacity of the battery.
    # Since the true aged capacity is available in the 'Capacity_initialized' column, the 'SOH' can be calculated by
    # simply dividing the 'Capacity_initialized' column by the nominal capacity value of the battery.
    # The 'SOH' is the target value for the Neural Network model to predict.
    df['SOH'] = df['Capacity_initialized'] / nominal_capacity

    # Re-sort the DataFrame after merging to maintain the correct order
    df = df.sort_values(['dataset_idx', 'Time'])

    # Now save the cudf Dataframe to a parquet file in the modified directory
    df.to_parquet(parquet_file_modified_path, index=False)

    # Save the first 100,002 rows to a CSV file
    df.iloc[:100002].to_csv(output_csv_path, index=False)

    print(f"Processed DataFrame with new column saved to {output_csv_path}")