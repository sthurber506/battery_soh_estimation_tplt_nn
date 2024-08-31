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
import shutil
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
import itertools

from IPython.display import display

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
    path_simulink_model = os.path.join(path_simulink_data_gen, 'simulink_model')
    path_simulink_results = os.path.join(path_simulink_data_gen, 'simulink_results')

    path_temp_slices = os.path.join(path_HPPC_interpolated_params, 'temp_slices')

    # Create the directories if they do not exist, but if they do exist, do not overwrite them
    os.makedirs(path_simulink_data_gen, exist_ok=True)
    os.makedirs(path_base_battery_model_params, exist_ok=True)
    os.makedirs(path_HPPC_base_params, exist_ok=True)
    os.makedirs(path_HPPC_interpolated_params, exist_ok=True)
    os.makedirs(path_temp_slices, exist_ok=True)
    os.makedirs(path_matlab_scripts, exist_ok=True)
    os.makedirs(path_simulink_model, exist_ok=True)
    os.makedirs(path_simulink_results, exist_ok=True)

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
        "simulink_results": path_simulink_results        
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

















































# Load the AHC_nom and SOC_vec_sim values from .mat files in the base battery model parameters directory
# Construct the full path to the .mat files
AHC_nom_file_path = os.path.join(paths["base_battery_model_params"], "AHC_nom.mat")
SOC_vec_sim_file_path = os.path.join(paths["base_battery_model_params"], "SOC_vec_sim.mat")

# Load the .mat files
AHC_nom_mat = scipy.io.loadmat(AHC_nom_file_path)
SOC_vec_sim_mat = scipy.io.loadmat(SOC_vec_sim_file_path)

# Extract AHC_nom as a single value
# Assuming 'AHC_nom' is the variable name inside the .mat file
AHC_nom = AHC_nom_mat['AHC_nom'].item()

# Extract SOC_vec_sim as a 1D array
# Assuming 'SOC_vec_sim' is the variable name inside the .mat file
SOC_vec_sim = SOC_vec_sim_mat['SOC_vec_sim'].squeeze()

# Define the parameters for the two-pulse test for Vmax simulation
# Define the percentage increment for capacity fade steps
percentage_increment = 2.5 # 2.5% increment

#Define the percentage start and end values for capacity fade
percentage_start = 100
percentage_end = 5

# Calculate the number of points needed for the range from percentage_start% to percentage_end% with the specified increment
# The formula used is: Number of points = ((End - Start) / Increment) + 1
# Convert percentage_increment to a fraction for calculation
num_points_capacity_fade = round(((percentage_start - percentage_end) / percentage_increment) + 1)

# Define a range for capacity fade values from percentage_start% to percentage_end% of initial capacity
# in specified percentage_increment steps using the AHC_nom value from the battery block
# and an equivalent of a MATLAB linspace function with the calculated number of points.
capacity_available_range = np.linspace(percentage_start, percentage_end, num_points_capacity_fade) * AHC_nom / 100

# Define the SOC range
SOC_range = SOC_vec_sim[SOC_vec_sim >= 0.11]

# Set the C-rate range maximum value for the simulation
c_rate_range_max = 1

# Set the C-rate range minimum value for the simulation
c_rate_range_min = 0.05

# Define the C-rate range steps wanted for the simulation
c_rate_range_steps = 0.05

# Correctly calculate the number of points for linspace
num_points_c_rate = round((c_rate_range_max - c_rate_range_min) / c_rate_range_steps) + 1

# Define the C-rate range for the simulation from 0.05 to 1 C-rate by using an equivalent of a MATLAB linspace function.
c_rate_range = np.linspace(c_rate_range_min, c_rate_range_max, num_points_c_rate)

# Calculate the load current for each C-rate value
pulse_current = c_rate_range * AHC_nom

# Define other parameters
sim_time = '50'
pulse_start = [10, 30]
pulse_duration = 10

# Generate all combinations of SOC, capacity, and C-rate values
all_combinations = list(itertools.product(SOC_range, capacity_available_range, c_rate_range))

# Total number of simulations
total_simulations = len(all_combinations)

# Display each variable calculated so far in a separate display
print("AHC_nom:", AHC_nom)
display(pd.DataFrame({"total_simulations": total_simulations}, index=[0]).style.set_properties(**{'text-align': 'left'}))
# display(pd.DataFrame({"all_combinations": all_combinations}).style.set_properties(**{'text-align': 'left'}))

# display(pd.DataFrame({"



# Define the weights for each parameter
weights = {
    "CPU_Threads": 1,
    "BoostClock": 50,
    "RAM": 1
}

# Calculate the computational power for each worker
worker_computational_power = {}
for worker_ip, capabilities in worker_capabilities.items():
    computational_power = (
        weights["CPU_Threads"] * capabilities["CPU_Threads"] +
        weights["BoostClock"] * capabilities["BoostClock"] +
        weights["RAM"] * capabilities["RAM"]
    )
    worker_computational_power[worker_ip] = computational_power

# Calculate the total computational power
total_computational_power = sum(worker_computational_power.values())

# Determine the number of simulations per worker based on computational power
worker_simulation_allocation = {
    worker_ip: round(total_simulations * (power / total_computational_power))
    for worker_ip, power in worker_computational_power.items()
}

# Adjust allocation to ensure total simulations are covered
allocated_simulations = sum(worker_simulation_allocation.values())
if allocated_simulations < total_simulations:
    worker_simulation_allocation[list(worker_simulation_allocation.keys())[0]] += (total_simulations - allocated_simulations)

# Print the simulation allocation
print("Worker Simulation Allocation:")
for worker_ip, sims in worker_simulation_allocation.items():
    print(f"{worker_ip}: {sims} simulations")
# Print a newline
print()







# Assuming all_combinations is defined and populated with all possible combinations
# Assuming worker_simulation_allocation is correctly populated with the number of simulations per worker

# Corrected distribution logic
start_idx = 0
worker_simulation_ranges = {}

for worker_ip, num_simulations in worker_simulation_allocation.items():
    end_idx = start_idx + num_simulations
    # Ensure the slice does not exceed the list bounds
    worker_simulation_ranges[worker_ip] = all_combinations[start_idx:min(end_idx, len(all_combinations))]
    start_idx = end_idx

# Print the simulation ranges for each worker
for worker_ip, simulations in worker_simulation_ranges.items():
    print(f"Worker {worker_ip}: {len(simulations)} simulations")
# Print a newline
print()


# Save worker_simulation_ranges to a CSV file
def save_worker_simulation_ranges(worker_simulation_ranges, file_path):
    data = []
    for worker_ip, simulations in worker_simulation_ranges.items():
        for sim in simulations:
            data.append([worker_ip] + list(sim))
    
    df = pd.DataFrame(data, columns=["worker_ip", "SOC", "capacity", "c_rate"])
    df.to_csv(file_path, index=False)

# Example usage
save_worker_simulation_ranges(worker_simulation_ranges, "/mnt/shared/simulink_data_generator/worker_simulation_ranges.csv")


















































import pandas as pd
import itertools
from dask.distributed import Client, wait
import subprocess
import os

def generate_matlab_params(arr):
    return "[" + " ".join(map(str, arr)) + "]"

# Correctly formatted MATLAB command and path variables
matlab_executable = "/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe"
matlab_scripts_path = "/mnt/shared/simulink_data_generator/matlab_scripts"
two_pulse_test_for_vmax_function_name = "python_dask_two_pulse_test_for_vmax"
local_simulink_model_path = "C:\\Users\\sthur\\Downloads\\SimulinkModels\\two_pulse_test_for_vmax_for_python.slx"
sim_time = '50'
pulse_start = [10, 30]
pulse_duration = 10

# Define Dask client connection
client = Client("tcp://100.82.123.17:8786")

# Save worker_simulation_ranges to a CSV file
def save_worker_simulation_ranges(worker_simulation_ranges, file_path):
    data = []
    for worker_ip, simulations in worker_simulation_ranges.items():
        for sim in simulations:
            data.append([worker_ip] + list(sim))
    
    df = pd.DataFrame(data, columns=["worker_ip", "SOC", "capacity", "c_rate"])
    df.to_csv(file_path, index=False)


save_worker_simulation_ranges(worker_simulation_ranges, "/mnt/shared/simulink_data_generator/worker_simulation_ranges.csv")

def submit_matlab_task(worker_ip, start_idx, end_idx, AHC_nom):
    # Adjust indices to match MATLAB's 1-based indexing
    start_idx += 1
    end_idx = min(end_idx + 1, len(worker_simulation_ranges))

    matlab_command = (
        f"\"{matlab_executable}\" -batch "
        f"\"addpath('{matlab_scripts_path}'); "
        f"{two_pulse_test_for_vmax_function_name}("
        f"'{local_simulink_model_path}', '{sim_time}', "
        f"{start_idx}, {end_idx}, "
        f"{generate_matlab_params(pulse_start)}, {pulse_duration}, 'LocalProfile1', {AHC_nom})\""
    )
    return client.submit(subprocess.run, matlab_command, shell=True, workers=[worker_ip])

# Submit tasks to each worker
futures = []
start_idx = 0
for worker_ip, num_simulations in worker_simulation_allocation.items():
    end_idx = start_idx + num_simulations - 1
    futures.append(submit_matlab_task(worker_ip, start_idx, end_idx, AHC_nom))
    start_idx = end_idx + 1

# Wait for all tasks to complete
wait(futures)

# Print the results
print("All MATLAB tasks completed.")

