import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
import pandas as pd
import numpy as np
import os
import gc
import gzip
import copy
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import random
import sys
import torch.nn.functional as F
import json
import math
from torch.utils.data.dataloader import default_collate
import tempfile
import shutil

# Create a temporary directory for normalization scalers
temp_dir = tempfile.mkdtemp()



class ELiSH(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x * torch.sigmoid(x), (torch.exp(x) - 1) * torch.sigmoid(x))

class HardELiSH(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x * torch.sigmoid(x), torch.sigmoid(x))

class GrowingCosineUnit(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)

class ShiftedQuadraticUnit(nn.Module):
    def forward(self, x):
        return x**2 + x

class DecayingSineUnit(nn.Module):
    def forward(self, x):
        return (np.pi / 2) * (torch.sinc(x - np.pi) - torch.sinc(x + np.pi))

class NonMonotonicCubicUnit(nn.Module):
    def forward(self, x):
        return x - x**3
    
def lecun_normal_(tensor):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = 1. / math.sqrt(fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


wandb.require("service")

print("Starting script...")

# Function to clear relevant memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_memory()

# Function to get the number of CPU cores
def get_num_workers():
    try:
        num_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        num_workers = multiprocessing.cpu_count()
    
    # if num_workers > 1:
    #     num_workers = 1  # Explicitly set to 1 if only one core is available

    print(f"Number of CPU cores available: {num_workers}")

    return num_workers

# Define a custom batch sampler to combine dataset_idx groups into batches based on random batch size
class GroupBatchSampler(Sampler):
    def __init__(self, subset, batch_size=1):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.group_indices = self._get_group_indices()
        self.batch_size = batch_size
    
    def _get_group_indices(self):
        group_indices = defaultdict(list)
        for subset_idx, dataset_idx in enumerate(self.indices):
            group = self.dataset.dataset_idx[dataset_idx].item()
            group_indices[group].append(subset_idx)
        return list(group_indices.values())

    def __iter__(self):
        np.random.shuffle(self.group_indices)
        for i in range(0, len(self.group_indices), self.batch_size):
            batch_groups = self.group_indices[i:i + self.batch_size]
            batch_indices = [idx for group in batch_groups for idx in group]
            yield batch_indices

    def __len__(self):
        return (len(self.group_indices) + self.batch_size - 1) // self.batch_size



# Define the dataset class
class BatteryDataset(Dataset):
    def __init__(self, parquet_dir, tensor_dir, input_cols, target_col='SOH', groups_per_file=11000):
        self.scalers = {}  # Add this line to store scalers
        self.tensor_files = [f for f in os.listdir(tensor_dir) if f.startswith('combined_') and f.endswith('.pt.gz')]
        self.input_cols = input_cols
        self.target_col = target_col
        self.tensor_dir = tensor_dir
        self.features_list, self.targets_list, self.dataset_idx_list = [], [], []
        self.groups_per_file = groups_per_file
        self.dataset_sizes = []
        self.num_input_features = len(input_cols)  # Store the number of input features
        self.max_dataset_idx = None  # Initialize max_dataset_idx        

        if self.tensor_files:
            print(f"Found {len(self.tensor_files)} saved tensor files.")
        else:
            print("No saved tensor files found.")

        if not self.tensor_files:
            combined_files = [f for f in os.listdir(parquet_dir) if f.startswith('combined_') and f.endswith('.parquet')]
            if combined_files:
                self.load_and_save_combined_files(combined_files, parquet_dir, tensor_dir)
            else:
                self.process_individual_files(parquet_dir)
                combined_files = [f for f in os.listdir(parquet_dir) if f.startswith('combined_') and f.endswith('.parquet')]
                self.load_and_save_combined_files(combined_files, parquet_dir, tensor_dir)
        else:
            self.infer_dataset_sizes()

            # Load tensors onto GPU sequentially
            self.load_tensors_onto_gpus()                        
            # self.load_tensors_in_chunks()

    def process_and_save_batch(self, batch_files, output_file):
        batch_data = []
        for i, file in enumerate(batch_files):
            print(f"Reading file {file}")
            df = pd.read_parquet(file)
            print(f"Appending data from file {file}")
            batch_data.append(df)

        print("Concatenating batch data")
        batch_data = pd.concat(batch_data).sort_values(['dataset_idx', 'Time']).reset_index(drop=True)
        print("Saving batch data to parquet file")
        batch_data.to_parquet(output_file)
        print(f"Saved batch data to {output_file}")
        clear_memory()

    def load_and_save_combined_files(self, combined_files, parquet_dir, tensor_dir):
        print(f"Found {len(combined_files)} combined parquet files. Loading and saving tensors...")

        num_workers = get_num_workers()

        def process_file(file):
            tensor_save_path = os.path.join(tensor_dir, file.replace('.parquet', '.pt.gz'))
            if os.path.exists(tensor_save_path):
                print(f"Found saved tensor at {tensor_save_path}. Loading...")
                with gzip.open(tensor_save_path, 'rb') as f:
                    features, targets, dataset_idx = torch.load(f)
            else:
                print(f"Reading file {file}")
                df = pd.read_parquet(os.path.join(parquet_dir, file))
                print(f"Appending data from file {file}")
                features = torch.tensor(df[self.input_cols].values, dtype=torch.float32)
                targets = torch.tensor(df[self.target_col].values, dtype=torch.float32)
                dataset_idx = torch.tensor(df['dataset_idx'].values, dtype=torch.float32)
                print(f"Saving tensors to {tensor_save_path}")
                with gzip.open(tensor_save_path, 'wb') as f:
                    torch.save((features, targets, dataset_idx), f)
                print(f"Saved tensors to {tensor_save_path}")
            return dataset_idx.unique().size(0)  # Number of unique dataset_idx groups

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            self.dataset_sizes = list(tqdm(executor.map(process_file, combined_files), total=len(combined_files), desc="Processing combined files"))

        clear_memory()
        print("All combined files have been processed and saved as tensors.")

    def load_tensors_in_chunks(self):
        print("Loading tensor files in chunks to avoid memory issues...")
        chunk_size = 2  # Adjust this value based on your system's memory capacity
        for i in range(0, len(self.tensor_files), chunk_size):
            self.load_chunk(i, i + chunk_size)

    def load_chunk(self, start_idx, end_idx):
        for file_idx in range(start_idx, end_idx):
            if file_idx < len(self.tensor_files):
                tensor_file = self.tensor_files[file_idx]
                with gzip.open(os.path.join(self.tensor_dir, tensor_file), 'rb') as f:
                    print(f"Loading tensor file {tensor_file}")
                    features, targets, dataset_idx = torch.load(f)
                    print(f"Loaded tensor file {tensor_file}")

                    print(f"Appending tensors from file {tensor_file}")
                    self.features_list.append(features)
                    print(f"Appended features from file {tensor_file}")
                    self.targets_list.append(targets)
                    print(f"Appended targets from file {tensor_file}")
                    self.dataset_idx_list.append(dataset_idx)
                    print(f"Appended dataset_idx from file {tensor_file}")

        if start_idx == 0:  # Only concatenate on the first load to ensure consistency
            print("Concatenating tensors...")
            self.features = torch.cat(self.features_list)
            print("Concatenated features.")
            self.targets = torch.cat(self.targets_list)
            print("Concatenated targets.")
            self.dataset_idx = torch.cat(self.dataset_idx_list)
            print("Concatenated dataset_idx.")
            print("All tensors concatenated.")

    def infer_dataset_sizes(self):
        # All files except the last one are assumed to have a fixed number of groups
        for _ in self.tensor_files[:-1]:
            self.dataset_sizes.append(self.groups_per_file)
        print(f"Assuming {self.groups_per_file} groups per file for all files except the last one.")
        
        # Print the dataset sizes for all files except the last one
        print("Inferred dataset sizes for all files except the last one:")
        for i, size in enumerate(self.dataset_sizes):
            print(f"File {i}: {size} groups")

        # Check if the actual dataset size for the last file is already known, and if so, load it
        last_file_size_file = os.path.join(self.tensor_dir, 'last_file_size.txt')
        if os.path.exists(last_file_size_file):
            with open(last_file_size_file, 'r') as size_file:
                last_file_size = int(size_file.read())
            print(f"Found actual dataset size for the last file: {last_file_size}")
            self.dataset_sizes.append(last_file_size)
        else:
            # Load the last tensor file to get its actual size
            if self.tensor_files:
                last_tensor_file = self.tensor_files[-1]
                print(f"Loading last tensor file {last_tensor_file} to infer dataset size...")
                with gzip.open(os.path.join(self.tensor_dir, last_tensor_file), 'rb') as f:
                    _, _, dataset_idx = torch.load(f)
                    last_file_size = dataset_idx.unique().size(0)
                    print(f"File {len(self.dataset_sizes)}: {last_file_size} groups")
                    
                    # Save the actual dataset size for the last file in a simple text file so that it can be used next time to avoid this step
                    with open(last_file_size_file, 'w') as size_file:
                        size_file.write(str(last_file_size))

                    # Print a message that the size of the last dataset has been saved to a text file
                    print(f"Saved actual dataset size for the last file in {last_file_size_file}")

                    print("Appending actual dataset size for the last file")
                    self.dataset_sizes.append(last_file_size)
                
        print("Inferred dataset sizes for all files:")
        for i, size in enumerate(self.dataset_sizes):
            print(f"File {i}: {size} groups")
                
    def process_individual_files(self, parquet_dir):
        print("Loading data from individual parquet files in batches...")
        parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        batch_size = 11  # Process 11 files at a time, you can adjust this as needed
        for start_idx in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[start_idx:start_idx + batch_size]
            output_file = os.path.join(parquet_dir, f"combined_{start_idx//batch_size}.parquet")
            self.process_and_save_batch(batch_files, output_file)

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.dataset_idx[idx]
    
    

    def load_tensors_onto_gpus(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError("No GPUs available for loading tensors.")

        all_features = []
        all_targets = []
        all_dataset_idx = []

        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        # Set max_split_size_mb to avoid memory fragmentation
        for device in devices:
            torch.cuda.memory.set_per_process_memory_fraction(0.8, device)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.cuda.empty_cache()

        def process_chunk(features_chunk, targets_chunk, dataset_idx_chunk, device):
            features_chunk = features_chunk.to(device)
            targets_chunk = targets_chunk.to(device)
            dataset_idx_chunk = dataset_idx_chunk.to(device)

            # Apply time shift and row deletion ONLY to simulated data (not real data)
            if tensor_file.startswith('combined_simulated_'):
                modified_features, modified_dataset_idx, modified_targets = apply_time_shift_and_row_deletion(features_chunk, dataset_idx_chunk, targets_chunk, self.input_cols)
            else:
                modified_features = features_chunk
                modified_dataset_idx = dataset_idx_chunk
                modified_targets = targets_chunk

            return modified_features.cpu(), modified_targets.cpu(), modified_dataset_idx.cpu()

        # Process each tensor file in parallel across multiple GPUs
        for tensor_file in self.tensor_files:
            print(f"Loading tensor file {tensor_file} onto GPUs")

            # Load the tensor file in chunks to avoid out-of-memory issues
            with gzip.open(os.path.join(self.tensor_dir, tensor_file), 'rb') as f:
                features, targets, dataset_idx = torch.load(f)

            num_rows = features.shape[0]
            group_size = 50001  # One dataset_idx group size
            chunk_size = 2500 * group_size  # 2500 dataset_idx groups

            # Split the data into chunks and distribute them across GPUs
            for i in range(0, num_rows, chunk_size):
                chunk_splits = np.array_split(range(i, min(i + chunk_size, num_rows)), num_gpus)
                processed_results = []

                for device, chunk_indices in zip(devices, chunk_splits):
                    chunk_features = features[chunk_indices]
                    chunk_targets = targets[chunk_indices]
                    chunk_dataset_idx = dataset_idx[chunk_indices]

                    processed_results.append(process_chunk(chunk_features, chunk_targets, chunk_dataset_idx, device))

                for modified_features, modified_targets, modified_dataset_idx in processed_results:
                    all_features.append(modified_features)
                    all_targets.append(modified_targets)
                    all_dataset_idx.append(modified_dataset_idx)

                torch.cuda.empty_cache()

        # Concatenate all processed data on the CPU
        self.features = torch.cat(all_features, dim=0)
        self.targets = torch.cat(all_targets, dim=0)
        self.dataset_idx = torch.cat(all_dataset_idx, dim=0)

        # Ensure the real_targets and self.targets have the same dimensions
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)  # Add a dimension to make it (N, 1)

        # **FIX:** Set max_dataset_idx after loading all simulated data
        self.max_dataset_idx = self.dataset_idx.max().item() + 1  # Save max_dataset_idx as a class attribute

        print("All tensors have been loaded, processed, and concatenated on the CPU.")

        # Load and process real data (ensure this is done after max_dataset_idx is set)
        real_data_tensor_file = 'real_data_tensor.pt'  # Modify with the correct file path
        print(f"Loading real data tensor from {real_data_tensor_file}")

        # Load the real data tensor and unpack it
        real_features, real_targets, real_dataset_idx = torch.load(os.path.join(self.tensor_dir, real_data_tensor_file))

        # Modify dataset_idx of real data to avoid overlap with simulated data
        real_dataset_idx += self.max_dataset_idx

        # Ensure the real_targets and self.targets have the same dimensions
        if len(real_targets.shape) == 1:
            real_targets = real_targets.unsqueeze(1)  # Add a dimension to make it (N, 1)

        # Concatenate real data with the simulated data
        self.features = torch.cat([self.features, real_features.cpu()], dim=0)
        self.targets = torch.cat([self.targets, real_targets.cpu()], dim=0)
        self.dataset_idx = torch.cat([self.dataset_idx, real_dataset_idx.cpu()], dim=0)

        print("Real data has been loaded, processed, and concatenated with simulated data.")




















# Custom collate function for dataset_idx groups with varying row sizes
def custom_collate_fn(batch):
    features, targets, dataset_idx = zip(*batch)
    return features[0], targets[0], dataset_idx[0]  # Return as is, without stacking
















# Modified apply_normalization function
def apply_normalization(dataset, method, scaler=None):
    # Check if dataset is a Subset and get the original dataset
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_size = 50001
    chunk_size = 10000 * group_size

    if scaler is not None:
        print(f"Using provided scaler for {method} normalization...")
        original_dataset.scaler = scaler
    else:
        if method == 'min-max':
            print("Applying Min-Max Normalization...")
            scaler = MinMaxScaler()
        elif method == 'std':
            print("Applying Standard Deviation Normalization...")
            scaler = StandardScaler()
        else:
            print("No normalization method applied.")
            return

        for i in range(0, len(original_dataset.features), chunk_size):
            chunk = original_dataset.features[i:i + chunk_size].to(device)
            scaler.partial_fit(chunk.cpu().numpy())

        original_dataset.scaler = scaler
        # Save the fitted scaler in the scalers dictionary
        original_dataset.scalers[method] = scaler

    normalized_features = []
    for i in range(0, len(original_dataset.features), chunk_size):
        chunk = original_dataset.features[i:i + chunk_size].to(device)
        normalized_chunk = original_dataset.scaler.transform(chunk.cpu().numpy())
        normalized_features.append(torch.tensor(normalized_chunk, dtype=torch.float32).to(device))

    original_dataset.features = torch.cat(normalized_features).cpu()

    print(f"{method.capitalize()} Normalization applied.")













# Function to reverse normalization
def reverse_normalization(dataset, scaler_path=None):
    # Check if the scaler exists in the dataset
    if hasattr(dataset, 'scaler') and dataset.scaler is not None:
        print("Using existing scaler for reverse normalization.")
    elif scaler_path is None or not os.path.exists(scaler_path):
        print("Scaler path is invalid or does not exist. Cannot reverse normalization.")
        return
    else:
        print(f"Loading scaler from {scaler_path} for reverse normalization...")
        dataset.scaler = torch.load(scaler_path)

    group_size = 50001  # One dataset_idx group size
    chunk_size = 10000 * group_size  # 10000 dataset_idx groups

    # Reverse the normalization in chunks
    reversed_features = []
    for i in range(0, len(dataset.features), chunk_size):
        chunk = dataset.features[i:i + chunk_size]
        reversed_chunk = dataset.scaler.inverse_transform(chunk.numpy())
        reversed_features.append(torch.tensor(reversed_chunk, dtype=torch.float32))

    # Concatenate the reversed features to update the dataset
    dataset.features = torch.cat(reversed_features)

    print("Normalization reversed.")

    # Clear the scaler from the dataset to free up memory
    del dataset.scaler
    torch.cuda.empty_cache()








def apply_time_shift_and_row_deletion(features, dataset_idx, targets, input_cols, max_shift=100000.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Applying Time Shift and Row Deletion...")

    # Move features, targets, and dataset_idx to the device (GPU or CPU)
    features = features.to(device)
    dataset_idx = dataset_idx.to(device)
    targets = targets.to(device)
    
    # Calculate dI/dt for each dataset_idx group
    current_idx = input_cols.index('Current')
    time_idx = input_cols.index('Time')

    # Compute dI/dt for the entire dataset (GPU vectorized)
    dI = torch.diff(features[:, current_idx], dim=0, prepend=torch.tensor([0.0], device=device))
    dT = torch.diff(features[:, time_idx], dim=0, prepend=torch.tensor([0.0], device=device))
    dI_dt = dI / torch.where(dT == 0, torch.ones_like(dT), dT)  # Avoid division by zero
    
    # Get random shift values for each dataset_idx group
    unique_dataset_idxs = dataset_idx.unique()
    shift_values = torch.FloatTensor(len(unique_dataset_idxs)).uniform_(0, max_shift).to(device)
    
    all_features = []
    all_targets = []
    all_dataset_idx = []

    for i, unique_idx in enumerate(unique_dataset_idxs):
        group_mask = dataset_idx == unique_idx
        group_features = features[group_mask]
        group_targets = targets[group_mask]  # Get the corresponding targets for this group


        # # For every 1000 groups, print the first 3 rows of all input columns
        # if i % 1000 == 0:
        #     print(f"\nFirst 3 rows of input columns for dataset_idx group {unique_idx.item()}:")
        #     print(f"{' | '.join(input_cols)}")
        #     for row in group_features[:3].cpu().numpy():
        #         formatted_row = [f"{row[time_idx]:.3f}" if idx == time_idx else f"{val:.5f}" for idx, val in enumerate(row)]
        #         print(" | ".join(formatted_row))
        
        # Time shift for the entire group
        group_features[:, time_idx] += shift_values[i]

        # Row deletion for each dataset_idx group
        group_dI_dt = dI_dt[group_mask]

        # Ensure no rows with non-zero dI/dt are deleted
        deletable_rows = torch.where(group_dI_dt == 0)[0]

        # Randomly choose row deletion strategy for this group
        row_deletion_option = random.choice(['None', '100Hz', '10Hz', '1Hz', 'Randomized'])
                
        # # For every 1000 groups, print the row deletion strategy for the current group
        # if i % 1000 == 0:
        #     print(f"\nRow deletion strategy for dataset_idx group {unique_idx.item()}: {row_deletion_option}")

        if row_deletion_option == 'None':
            # No rows are deleted, retain the entire group
            all_features.append(group_features)
            all_targets.append(group_targets)
            all_dataset_idx.append(dataset_idx[group_mask])
            continue

        elif row_deletion_option == '100Hz':
            # Delete rows to simulate 100Hz sampling (keep every 10th row)
            keep_mask = torch.zeros_like(deletable_rows, dtype=torch.bool, device=device)
            keep_mask[::10] = True  # Keep every 10th row

        elif row_deletion_option == '10Hz':
            # Delete rows to simulate 10Hz sampling (keep every 100th row)
            keep_mask = torch.zeros_like(deletable_rows, dtype=torch.bool, device=device)
            keep_mask[::100] = True  # Keep every 100th row

        elif row_deletion_option == '1Hz':
            # Delete rows to simulate 1Hz sampling (keep every 1000th row)
            keep_mask = torch.zeros_like(deletable_rows, dtype=torch.bool, device=device)
            keep_mask[::1000] = True  # Keep every 1000th row

        elif row_deletion_option == 'Randomized':
            # Random deletion, but ensure no more rows are deleted than the 1Hz strategy would delete
            max_deletions = len(deletable_rows) - len(deletable_rows) // 1000  # At least keep 1Hz equivalent rows
            num_keep = len(deletable_rows) - random.randint(0, max_deletions)
            keep_indices = torch.randperm(len(deletable_rows))[:num_keep]
            keep_mask = torch.zeros_like(deletable_rows, dtype=torch.bool, device=device)
            keep_mask[keep_indices] = True  # Mark the rows to keep

        # Apply the keep mask and collect the remaining rows
        kept_rows = deletable_rows[keep_mask]
        all_features.append(group_features[kept_rows])
        all_targets.append(group_targets[kept_rows])
        all_dataset_idx.append(dataset_idx[group_mask][kept_rows])

        # # Print the first 3 rows of all input columns for every 1000 groups
        # if i % 1000 == 0:
        #     print(f"\nAfter row deletion - First 3 rows of input columns for dataset_idx group {unique_idx.item()}:")
        #     for row in group_features[kept_rows][:3].cpu().numpy():
        #         formatted_row = [f"{row[time_idx]:.3f}" if idx == time_idx else f"{val:.5f}" for idx, val in enumerate(row)]
        #         print(" | ".join(formatted_row))

        #     print(f"Number of rows before deletion: {len(group_features)}")
        #     print(f"Number of rows after deletion: {len(kept_rows)}")
        #     print("-" * 40)

    # Concatenate all remaining features, targets, and dataset_idx back together
    features = torch.cat(all_features, dim=0).cpu()  # Ensure features are moved back to the CPU
    targets = torch.cat(all_targets, dim=0).cpu()    # Ensure targets are moved back to the CPU
    dataset_idx = torch.cat(all_dataset_idx, dim=0).cpu()  # Ensure dataset_idx is moved back to the CPU

    print("Time shift and row deletion applied successfully.")
    return features, dataset_idx, targets


































# Columns to be used as input features for the neural network
input_cols = [
    'Time', 'Voltage', 'Current', 'SOC', 'Capacity_initialized',
    'C_rate_initialized', 'OCV', 'tplt_V_max', 'tplt_delta_V_2',
    'C_rate_based_on_AHC_initialized'
]

# Define paths
parquet_dir = '/home/sthurber506/tplt/parquet_files'
tensor_dir = '/home/sthurber506/tplt/tensors'

# Initialize the dataset
dataset = BatteryDataset(parquet_dir, tensor_dir, input_cols)

# Get the number of workers
num_workers = get_num_workers()




# Updated function to create dataloaders with normalization and reversal, using batch_size to control the number of dataset_idx groups per batch
def create_normalized_dataloaders(normalization_method, batch_size, local_dataset):
    # Get unique dataset_idx groups
    all_unique_dataset_idxs = torch.unique(local_dataset.dataset_idx).numpy()

    # Separate real and simulated data by dataset_idx value
    real_data_mask = all_unique_dataset_idxs >= local_dataset.max_dataset_idx
    simulated_data_mask = all_unique_dataset_idxs < local_dataset.max_dataset_idx

    real_dataset_idxs = all_unique_dataset_idxs[real_data_mask]
    simulated_dataset_idxs = all_unique_dataset_idxs[simulated_data_mask]

    np.random.shuffle(simulated_dataset_idxs)

    # Split the dataset based on dataset_idx groups
    train_size = int(0.7 * len(simulated_dataset_idxs))  # 70% for training
    val_size = int(0.15 * len(simulated_dataset_idxs))   # 15% for validation
    test_size = len(simulated_dataset_idxs) - train_size - val_size  # Remaining 15% for testing

    train_dataset_idxs = simulated_dataset_idxs[:train_size]
    val_dataset_idxs = simulated_dataset_idxs[train_size:train_size + val_size]
    test_dataset_idxs = np.concatenate([simulated_dataset_idxs[train_size + val_size:], real_dataset_idxs])  # Add more real data to test

    # Create masks to filter the dataset based on dataset_idx groups
    train_mask = torch.isin(local_dataset.dataset_idx, torch.tensor(train_dataset_idxs))
    val_mask = torch.isin(local_dataset.dataset_idx, torch.tensor(val_dataset_idxs))
    test_mask = torch.isin(local_dataset.dataset_idx, torch.tensor(test_dataset_idxs))

    # Subset the dataset
    train_dataset = Subset(local_dataset, torch.where(train_mask)[0])
    val_dataset = Subset(local_dataset, torch.where(val_mask)[0])
    test_dataset = Subset(local_dataset, torch.where(test_mask)[0])

    train_scaler = None  # Initialize train_scaler to None

    # Normalize the training set and fit the scaler
    if normalization_method != 'none':
        apply_normalization(train_dataset, normalization_method)  # Fit scaler on the training dataset
        train_scaler = local_dataset.scalers[normalization_method]  # Save the scaler after fitting it on the training data
        
        # Apply the fitted scaler to validation and test sets
        apply_normalization(val_dataset, normalization_method, scaler=train_scaler)
        apply_normalization(test_dataset, normalization_method, scaler=train_scaler)

    # Use the sweep-configured batch size to control how many dataset_idx groups are combined into each batch
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=num_workers, pin_memory=True, 
                            sampler=GroupBatchSampler(train_dataset, batch_size=batch_size), prefetch_factor=8, 
                            persistent_workers=True, collate_fn=custom_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, pin_memory=True, 
                            sampler=GroupBatchSampler(val_dataset, batch_size=batch_size), prefetch_factor=8, 
                            persistent_workers=True, collate_fn=custom_collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, pin_memory=True, 
                            sampler=GroupBatchSampler(test_dataset, batch_size=batch_size), prefetch_factor=8, 
                            persistent_workers=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader, train_scaler






















print("Initializing wandb...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device chosen:", device)

# Define the simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation_function=nn.ReLU(), output_activation_function=None, dropout=0.5):
        super(SimpleNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, 1)

        print("Model Architecture:")
        print(self)
        print()

    def forward(self, x):
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = self.activation_function(x)
                x = layer(x)

        x = self.output_layer(x)
        if self.output_activation_function is not None:
            x = self.output_activation_function(x)
        return x





def print_and_log_model_architecture(input_dim, hidden_dims, output_dim=1):
    architecture = []
    architecture.append(f"Input Layer: {input_dim} neurons")
    print(architecture[-1])
    
    layer_count = 1  # Counter for hidden layers, skipping dropout layers
    for i, layer in enumerate(hidden_dims):
        layer_info = f"Hidden Layer {layer_count}: {layer} neurons"
        architecture.append(layer_info)
        print(layer_info)
        layer_count += 1
    
    output_layer_info = f"Output Layer: {output_dim} neuron"
    architecture.append(output_layer_info)
    print(output_layer_info)

    # Log architecture to wandb
    # wandb.log({"model_architecture": "\n".join(architecture)})




def apply_weight_initialization(layer, activation_function, is_output_layer=False, layer_name="layer", weight_init_strategy='uniform', weight_init_choice=None):    
    # Check if the weight_init_strategy is 'uniform' and if weight_init_choice is not None.
    # If both conditions are met, use the weight_init_choice to apply the chosen weight initialization and then return.
    if weight_init_strategy == 'uniform' and weight_init_choice is not None:
        # Check if the weight_init_choice is 'default' and if it is, stick with PyTorch's default initialization.
        if weight_init_choice == 'default':
            print(f"Using default PyTorch initialization for {layer_name}")
            
            # Log the weight initialization strategy as 'default' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "Default (PyTorch)"}, allow_val_change=False)

            # Set init_choice to 'default'.
            init_choice = 'default'
            # Return the init_choice value.
            return init_choice
        
        # If the weight_init_choice is not 'default', apply the chosen weight initialization.
        elif weight_init_choice == 'xavier':
            print(f"Applying Xavier initialization to {layer_name}")
            torch.nn.init.xavier_uniform_(layer.weight)

            # Log the weight initialization strategy as 'Xavier' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "Xavier"}, allow_val_change=False)

        elif weight_init_choice == 'lecun':
            print(f"Applying LeCun initialization to {layer_name}")
            lecun_normal_(layer.weight)
            
            # Log the weight initialization strategy as 'LeCun' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "LeCun"}, allow_val_change=False)

        elif weight_init_choice == 'random':
            print(f"Applying Random initialization to {layer_name}")
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            
            # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

        elif weight_init_choice == 'he':
            print(f"Applying He initialization to {layer_name}")
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            
            # Log the weight initialization strategy as 'He' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "He"}, allow_val_change=False)
        
        # Set init_choice to weight_init_choice.
        init_choice = weight_init_choice

        # Return the init_choice value.
        return init_choice

    # Randomly decide whether to use custom initialization or stick with the PyTorch default
    use_custom_init = random.choice([True, False])
    
    if use_custom_init:
        if is_output_layer and isinstance(activation_function, nn.Sigmoid):
            # Random choice between Xavier, LeCun, or Random initialization for output layer
            init_choice = random.choice(['xavier', 'lecun', 'random'])
            if init_choice == 'xavier':
                print(f"Applying Xavier initialization to {layer_name}")
                torch.nn.init.xavier_uniform_(layer.weight)
                
                # Log the weight initialization strategy as 'Xavier' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Xavier"}, allow_val_change=False)

            elif init_choice == 'lecun':
                print(f"Applying LeCun initialization to {layer_name}")
                lecun_normal_(layer.weight)
                
                # Log the weight initialization strategy as 'LeCun' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "LeCun"}, allow_val_change=False)

            elif init_choice == 'random':
                print(f"Applying Random initialization to {layer_name}")
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                
                # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

        elif isinstance(activation_function, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
            # Random choice between He initialization and Random initialization for ReLU variants
            init_choice = random.choice(['he', 'random'])
            if init_choice == 'he':
                print(f"Applying He initialization to {layer_name}")
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                
                # Log the weight initialization strategy as 'He' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "He"}, allow_val_change=False)

            elif init_choice == 'random':
                print(f"Applying Random initialization to {layer_name}")
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                
                # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

        # Check if the activation function is GELU, and if it is, randomly choose between Xavier and Random initialization
        elif isinstance(activation_function, nn.GELU):
            # Random choice between Xavier and Random initialization for GELU
            init_choice = random.choice(['xavier', 'random'])
            if init_choice == 'xavier':
                print(f"Applying Xavier initialization to {layer_name}")
                torch.nn.init.xavier_uniform_(layer.weight)
                
                # Log the weight initialization strategy as 'Xavier' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Xavier"}, allow_val_change=False)

            elif init_choice == 'random':
                print(f"Applying Random initialization to {layer_name}")
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                
                # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

        # Check if the activation function is SELU, and if it is, randomly choose between LeCun initialization and Random initialization
        elif isinstance(activation_function, nn.SELU):
            # Random choice between LeCun and Random initialization for SELU
            init_choice = random.choice(['lecun', 'random'])
            if init_choice == 'lecun':
                print(f"Applying LeCun initialization to {layer_name}")
                lecun_normal_(layer.weight)
                
                # Log the weight initialization strategy as 'LeCun' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "LeCun"}, allow_val_change=False)

            elif init_choice == 'random':
                print(f"Applying Random initialization to {layer_name}")
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                
                # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
                wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

        # Otherwise, apply Random initialization for other activations
        else:
            # Apply Random initialization for other activations
            print(f"Applying Random initialization to {layer_name}")
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            
            # Log the weight initialization strategy as 'Random' for this layer and prevent further changes.
            wandb.config.update({f"{layer_name}_weight_init": "Random"}, allow_val_change=False)

            # Set init_choice to 'random'.
            init_choice = 'random'
    else:
        # Stick with PyTorch's default initialization
        print(f"Using default PyTorch initialization for {layer_name}")
        
        # Log the weight initialization strategy as 'default' for this layer and prevent further changes.
        wandb.config.update({f"{layer_name}_weight_init": "Default (PyTorch)"}, allow_val_change=False)

        # Set init_choice to 'default'.
        init_choice = 'default'

    # Return the init_choice value.
    return init_choice





def create_model(input_dim, config):
    activation_functions = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "gelu": nn.GELU(),
        "selu": nn.SELU(),
        "elish": ELiSH(),
        "hardelish": HardELiSH(),
        "gcu": GrowingCosineUnit(),
        "squ": ShiftedQuadraticUnit(),
        "dsu": DecayingSineUnit(),
        "ncu": NonMonotonicCubicUnit(),
        "sigmoid": nn.Sigmoid(),  # Added sigmoid activation function
    }

    activation_function = activation_functions[config.activation_function]
    output_activation_function = None if config.output_activation_function == "none" else activation_functions[config.output_activation_function]

    print(f"Chosen activation function: {config.activation_function}")
    print(f"Chosen output activation function: {config.output_activation_function}")
    
    hidden_dims = generate_hidden_dims(config)
    
    print_and_log_model_architecture(input_dim, hidden_dims)
    
    model = SimpleNN(input_dim, hidden_dims, activation_function, output_activation_function, config.dropout).to(device)

    # Create a way to randomly choose to apply weight initialization uniformly across all layers or individually.
    # Randomly choose between 'uniform' and 'individual' and store the result in a variable.
    # weight_init_strategy = random.choice(['uniform', 'individual'])
    weight_init_strategy = config.weight_init_strategy
    print(f"Weight Initialization Strategy: {weight_init_strategy}")

    # Initialize a variable to store the weight initialization choice that is returned from the apply_weight_initialization function.
    weight_init_choice = None

    # Initialize a variable to track the hidden layer count.
    hidden_layer_count = 1

    # Apply weight initialization to hidden layers based on activation function
    for i, layer in enumerate(model.hidden_layers):
        if isinstance(layer, nn.Linear):
            # Store the weight initialization choice in the weight_init_choice variable.
            weight_init_choice = apply_weight_initialization(layer, activation_function, layer_name=f"hidden_layer_{hidden_layer_count}", weight_init_strategy=weight_init_strategy, weight_init_choice=weight_init_choice)
            # Increment the hidden_layer_count variable by 1.
            hidden_layer_count += 1
    
    # Apply weight initialization for the output layer
    apply_weight_initialization(model.output_layer, output_activation_function, is_output_layer=True, layer_name="output_layer")


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU.")
    
    return model



def generate_hidden_dims(config):
    # hidden_layers_min = config.hidden_layers_min
    # hidden_layers_max = config.hidden_layers_max
    # hidden_layers = random.randint(hidden_layers_min, hidden_layers_max)
    hidden_layers = config.hidden_layers
    print(f"Number of Hidden Layers: {hidden_layers}")    

    absolute_dim_range_min = config.absolute_dim_range_min
    absolute_dim_range_max = config.absolute_dim_range_max

    # hidden_dim_initial_range_min = config.hidden_dim_initial_range_min
    # hidden_dim_initial_range_max = config.hidden_dim_initial_range_max

    # hidden_dim_random_scale_factor_range_min = config.hidden_dim_random_scale_factor_range_min
    # hidden_dim_random_scale_factor_range_max = config.hidden_dim_random_scale_factor_range_max
    
    scale_factor_fixed = config.hidden_dim_scale_factor_fixed
    randomize_strategy = config.randomize_hidden_dim_strategy

    hidden_dim_strategy = config.hidden_dim_strategy
    hidden_dims = []

    print(f"Scale Factor Fixed: {scale_factor_fixed}")
    print(f"Randomize Hidden Dim Strategy: {randomize_strategy}")
    print(f"Hidden Dim Strategy: {hidden_dim_strategy}")
    print()

    # current_dim = random.randint(hidden_dim_initial_range_min, hidden_dim_initial_range_max)
    current_dim = config.hidden_dim_initial
    hidden_dims.append(current_dim)
    print(f"Hidden Layer 1 Neurons: {current_dim}")
    # wandb.log({f"hidden_layer_1_neurons": current_dim})

    # hidden_dim_scale_factor = random.uniform(hidden_dim_random_scale_factor_range_min, hidden_dim_random_scale_factor_range_max)
    hidden_dim_scale_factor = config.hidden_dim_scale_factor
    print(f"Hidden Dim Random Scale Factor: {hidden_dim_scale_factor}")
    # log the hidden_dim_scale_factor
    # wandb.log({f"hidden_dim_scale_factor": hidden_dim_scale_factor})

    print(f"Range for Hidden Layers: {list(range(2, hidden_layers + 1))}")
    print()    

    for layer in range(2, hidden_layers + 1):
        if layer != 2 and randomize_strategy:
            hidden_dim_strategy = random.choice(['compress', 'expand', 'maintain'])
        
        if hidden_dim_strategy == 'maintain':
            hidden_dims.append(current_dim)

        else:
            if layer != 2 and not scale_factor_fixed:
                hidden_dim_scale_factor = random.uniform(1.01, 2.0)

            if hidden_dim_strategy == 'compress':
                current_dim = max(absolute_dim_range_min, int(current_dim / hidden_dim_scale_factor))

            elif hidden_dim_strategy == 'expand':
                current_dim = min(absolute_dim_range_max, int(current_dim * hidden_dim_scale_factor))
            
            hidden_dims.append(current_dim)

        # Log the hidden_layer_neurons for each layer and prevent further changes.
        wandb.config.update({f"hidden_layer_{layer}_neurons": current_dim}, allow_val_change=False)
        print(f"Hidden Layer {layer} Neurons: {current_dim}")

        # Log the hidden_dim_scale_factor for each layer and prevent further changes.
        wandb.config.update({f"hidden_layer_{layer}_scale_factor": hidden_dim_scale_factor}, allow_val_change=False)
        print(f"Hidden Dim Scale Factor for Layer {layer}: {hidden_dim_scale_factor}")

        # Log the hidden_dim_strategy for each layer and prevent further changes.
        wandb.config.update({f"hidden_layer_{layer}_strategy": hidden_dim_strategy}, allow_val_change=False)
        print(f"Hidden Dim Strategy for Layer {layer}: {hidden_dim_strategy}")


    print(f"Hidden Layer Neurons: {hidden_dims}")
    print()

    # wandb.config.actual_hidden_layers = hidden_layers

    return hidden_dims





# Early stopping parameters based on accuracy or loss
early_stop_acc_threshold = 0.99  # Set your desired accuracy threshold
early_stop_loss_threshold = 1 - early_stop_acc_threshold  # Equivalent loss threshold based on accuracy
early_stop_patience = 50  # Number of epochs to wait before stopping when the criteria are met



# Modified training loop with early stopping based on accuracy or loss
def train_model(train_loader, val_loader, test_loader, penalty_factor, apply_penalty):
    model.train()
    running_loss = 0.0
    total_train = 0

    best_val_loss = float('inf')  # Initialize best_val_loss
    best_combined_metric_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0
    patience = 20  # Number of epochs to wait for improvement before stopping
    early_stop_epochs = 0  # Counter for consecutive epochs above the early stop threshold
    threshold_accuracy = 0.85
    min_loss_decrease = 1e-3
    train_val_loss_history = []
    moving_avg_window = 15  # Window size for moving average

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total_train = 0

        for inputs, targets, dataset_idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(outputs.shape)  # Adjust target shape to match output
            loss = criterion(outputs, targets)

            # Apply penalty for outputs outside the range [0, 1]
            if apply_penalty:
                penalty = torch.sum(torch.maximum(torch.zeros_like(outputs), outputs - 1.0) + torch.maximum(torch.zeros_like(outputs), -outputs))
                loss += penalty_factor * penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train += targets.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 1 - np.sqrt(epoch_loss)  # Calculate accuracy from loss
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "train_accuracy": train_accuracy})

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Training Loss: {epoch_loss}, Training Accuracy: {train_accuracy}')

        model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.view(outputs.shape)  # Adjust target shape to match output
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                total_val += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 1 - np.sqrt(val_loss)  # Calculate accuracy from loss
        wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_accuracy": val_accuracy})

        # Calculate the train_val_loss and train_val_accuracy by averaging the train and validation loss and accuracy
        train_val_loss = (epoch_loss + val_loss) / 2
        train_val_accuracy = (train_accuracy + val_accuracy) / 2
        wandb.log({"epoch": epoch+1, "train_val_loss": train_val_loss, "train_val_accuracy": train_val_accuracy})

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

        # Early stopping condition based on accuracy or loss
        if train_accuracy >= early_stop_acc_threshold and val_accuracy >= early_stop_acc_threshold:
            early_stop_epochs += 1
            if early_stop_epochs >= early_stop_patience:
                print(f'Early stopping triggered at epoch {epoch+1} due to train and validation accuracy above {early_stop_acc_threshold}')
                break
        elif epoch_loss <= early_stop_loss_threshold and val_loss <= early_stop_loss_threshold:
            early_stop_epochs += 1
            if early_stop_epochs >= early_stop_patience:
                print(f'Early stopping triggered at epoch {epoch+1} due to train and validation loss below {early_stop_loss_threshold}')
                break
        else:
            early_stop_epochs = 0  # Reset the counter if the condition is not met

        # Update train_val_loss_history
        train_val_loss_history.append(train_val_loss)
        if len(train_val_loss_history) > moving_avg_window:
            train_val_loss_history.pop(0)

        # Calculate moving average of validation loss
        moving_avg_train_val_loss = sum(train_val_loss_history) / len(train_val_loss_history)

        # If the val loss is NaN and the epoch is greater than 10, break the loop
        if np.isnan(val_loss) and epoch > 10:
            print(f'Validation loss is NaN at epoch {epoch+1}. Stopping early.')
            break

        # Check for improvement
        if epoch >= moving_avg_window - 1:
            print(f'Moving Average Train_Validation Loss: {moving_avg_train_val_loss}')
            if moving_avg_train_val_loss < best_val_loss:
                if val_accuracy < threshold_accuracy and (best_val_loss - moving_avg_train_val_loss) < min_loss_decrease:
                    print(f'Stopping early due to slow decrease in train_validation loss at epoch {epoch+1}')
                    break
                best_val_loss = moving_avg_train_val_loss
                best_model_wts = model.state_dict().copy()
                epochs_no_improve = 0
                print(f"New best model found at epoch {epoch+1} with moving average train_validation loss {moving_avg_train_val_loss}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f'Early stopping triggered at epoch {epoch+1} due to no improvement in moving average train_validation loss for {patience} epochs')
                break

        # Implement a hard early stopping condition if training or validation accuracy are below 0.5 for more than 15 epochs.
        if epoch > 15:
            if train_accuracy < 0.5 or val_accuracy < 0.5:
                print(f'Early stopping triggered at epoch {epoch+1} due to train or validation accuracy below 0.5')
                break

        # Implement a hard early stopping condition if training or validation loss are below 0.8 for more than 75 epochs.
        if epoch > 75:
            if epoch_loss < 0.8 or val_loss < 0.8:
                print(f'Early stopping triggered at epoch {epoch+1} due to train or validation loss below 0.8')
                break

        

    print("Evaluating the model on the test set...")
    if best_model_wts:
        model.load_state_dict(best_model_wts)


    model.eval()
    test_loss = 0.0
    total_test = 0
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.view(outputs.shape)  # Adjust target shape to match output
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            total_test += targets.size(0)

    test_loss /= len(test_loader)
    test_accuracy = 1 - np.sqrt(test_loss)  # Calculate accuracy from loss
    combined_metric_loss = (epoch_loss + val_loss + test_loss) / 3  # Average of train, val, and test loss
    combined_metric_accuracy = (train_accuracy + val_accuracy + test_accuracy) / 3  # Average of train, val, and test accuracy
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "combined_metric_loss": combined_metric_loss, "combined_metric_accuracy": combined_metric_accuracy})
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Combined Metric Loss: {combined_metric_loss}, Combined Metric Accuracy: {combined_metric_accuracy}')

    if isinstance(model, nn.DataParallel):
        architecture = {"input_dim": model.module.input_dim, "hidden_dims": model.module.hidden_dims}
    else:
        architecture = {"input_dim": model.input_dim, "hidden_dims": model.hidden_dims}

    performance = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "combined_metric_loss": combined_metric_loss,
        "combined_metric_accuracy": combined_metric_accuracy
    }

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "combined_metric_loss": combined_metric_loss,
        "combined_metric_accuracy": combined_metric_accuracy,
        "model_state_dict": model.state_dict().copy(),
        "config": {k: v for k, v in config.items() if k != '_wandb'},  # Exclude non-pickleable items
        "architecture": architecture,
        "performance": performance,
        "normalization_method": config.normalization_method  # Save normalization method
    }



# Define the function to select the loss function
def get_loss_function(config):
    chosen_loss = config.loss_function

    if chosen_loss == "mse":
        print("Using MSE Loss")
        return nn.MSELoss().to(device)
    elif chosen_loss == "mae":
        print("Using MAE Loss")
        return nn.L1Loss().to(device)
    elif chosen_loss == "huber":
        delta = random.uniform(0.1, 2.0)  # Randomize delta between 0.1 and 2.0

        # Log delta for this run and prevent further changes
        wandb.config.update({"huber_delta": delta}, allow_val_change=False)

        print(f"Using Huber Loss with delta={delta}")
        
        return nn.HuberLoss(delta=delta).to(device)








input_dim = len(input_cols)

all_run_results = []

def sweep_train():
    retries = 3
    for attempt in range(retries):
        try:
            with wandb.init() as run:
                global config
                config = wandb.config

                # Reset any previous normalization
                reverse_normalization(dataset)

                apply_penalty = config.apply_penalty
                penalty_factor = config.penalty_factor if apply_penalty else 0.0

                if apply_penalty:
                    print(f"Applying penalty with factor {penalty_factor}")
                else:
                    print("Not applying penalty")

                global model
                model = create_model(input_dim, config)

                train_loader, val_loader, test_loader, scaler_path = create_normalized_dataloaders(config.normalization_method, config.batch_size, dataset)

                optimizer_options = {
                    "adam": optim.Adam,
                    "sgd": lambda params, lr: optim.SGD(params, lr=lr, momentum=config.momentum, nesterov=config.nesterov),
                    "rmsprop": optim.RMSprop,
                    "adamw": optim.AdamW,
                    "adadelta": optim.Adadelta,
                    "adamax": optim.Adamax,
                    "nadam": optim.NAdam,
                    "asgd": optim.ASGD
                }
                optimizer_class = optimizer_options[config.optimizer]
                global optimizer
                optimizer = optimizer_class(model.parameters(), lr=config.learning_rate)

                print(f"Optimizer: {config.optimizer}, Learning Rate: {config.learning_rate}")

                global criterion
                criterion = get_loss_function(config)

                run_result = train_model(train_loader, val_loader, test_loader, penalty_factor, apply_penalty)
                all_run_results.append((run_result, scaler_path))

                # Clear memory
                del train_loader, val_loader, test_loader
                torch.cuda.empty_cache()

                # Reverse normalization to reset the dataset
                reverse_normalization(dataset)

            break
        except Exception as e:
            print(f"Error encountered: {e}. Retrying {attempt + 1}/{retries}...")
            time.sleep(5)
    else:
        print("Failed to complete sweep after multiple attempts.")






# Function to save the best model and details
def save_model_and_details(best_run, model_filename, json_filename, entire_model_filename, model, scaler_path):
    model_details = {
        "input_columns": input_cols,
        "architecture": best_run['architecture'],
        "performance": {
            "test_loss": float(f"{best_run['test_loss']:.10f}"),
            "test_accuracy": float(f"{best_run['test_accuracy']:.10f}"),
            "combined_metric_loss": float(f"{best_run['combined_metric_loss']:.10f}"),
            "combined_metric_accuracy": float(f"{best_run['combined_metric_accuracy']:.10f}")
        },
        "config": best_run['config'],  # Save the entire config to JSON
        "apply_penalty": best_run['config'].get('apply_penalty', False),
        "penalty_factor": best_run['config'].get('penalty_factor', 0.0),
        "normalization_method": best_run['normalization_method']  # Save normalization method
    }
    # Save state dictionary
    torch.save({
        'model_state_dict': best_run['model_state_dict'],
        'config': best_run['config'],
        'architecture': best_run['architecture'],
        'performance': model_details['performance'],
        'combined_metric_loss': best_run['combined_metric_loss'],  # Ensure combined_metric_loss is included
    }, model_filename)
    # Save model details as JSON
    with open(json_filename, 'w') as f:
        json.dump(model_details, f, indent=4)
    print(f"Best model and details saved as {model_filename} and {json_filename}")

    # Unwrap the model if it is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    # Save the entire model
    model_scripted = torch.jit.script(model)
    model_scripted.save(entire_model_filename)
    print(f"Entire model saved as {entire_model_filename}")

    # Save scaler state if available
    if scaler_path:
        torch.save(torch.load(scaler_path), scaler_path)
        print(f"Scaler saved as {scaler_path}")









if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.1
            },
            'epochs': {
                'value': 10000
            },
            'batch_size': {
                'min': 10,
                'max': 50
            },
            # 'hidden_layers_min': {
            #     'value': 2
            # },
            # 'hidden_layers_max': {
            #     'value': 7
            'hidden_layers': {
                'min': 2,
                'max': 7
            },
            'absolute_dim_range_min': {
                'value': 4
            },
            'absolute_dim_range_max': {
                'value': 1024
            },
            # 'hidden_dim_initial_range_min': {
            #     'value': 8
            # },
            # 'hidden_dim_initial_range_max': {
            #     'value': 512
            'hidden_dim_initial': {
                'min': 8,
                'max': 512
            },
            # 'hidden_dim_random_scale_factor_range_min': {
            #     'value': 1.01
            # },
            # 'hidden_dim_random_scale_factor_range_max': {
            #     'value': 2.0
            'hidden_dim_scale_factor': {
                'min': 1.01,
                'max': 2.0
            },
            'hidden_dim_scale_factor_fixed': {
                'values': [True, False]
            },
            'randomize_hidden_dim_strategy': {
                'values': [True, False]
            },
            'hidden_dim_strategy': {
                'values': ['compress', 'expand', 'maintain']
            },
            'activation_function': {
                'values': ['relu', 'leaky_relu', 'elu', 'softplus', 'swish', 'mish', 'gelu', 'selu', 'elish', 'hardelish', 'gcu', 'squ', 'dsu', 'ncu']
            },
            'output_activation_function': {
                'values': ['none', 'sigmoid']
            },
            'dropout': {
                'min': 0.2,
                'max': 0.8
            },
            'normalization_method': {
                'values': ['none', 'min-max', 'std']
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop', 'adamw', 'adadelta', 'adamax', 'nadam', 'asgd']
            },
            'momentum': {
                'min': 0.0,
                'max': 0.9
            },
            'nesterov': {
                'values': [False, True]
            },
            'apply_penalty': {
                'values': [True, False]
            },
            # 'penalty_factor_min': {
            #     'value': 1.0
            # },
            # 'penalty_factor_max': {
            #     'value': 100.0
            # }
            'penalty_factor': {
                'min': 1.0,
                'max': 10.0
            },
            'weight_init_strategy': {
                'values': ['uniform', 'individual']
            },
            'loss_function': {
                'values': ['mse', 'mae', 'huber']
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='battery-soh-prediction')
    wandb.agent(sweep_id, function=sweep_train, count=300)

    if all_run_results:
        valid_runs = [run for run in all_run_results if not any(np.isnan(metric) for metric in [run[0]['test_loss'], run[0]['combined_metric_loss'], run[0]['combined_metric_accuracy']])]
        if valid_runs:
            best_run, best_scaler_path = min(valid_runs, key=lambda x: x[0]['combined_metric_loss'])
            print(f"Best model found with test loss {best_run['test_loss']}, accuracy {best_run['test_accuracy']}, combined metric loss {best_run['combined_metric_loss']}, and combined metric accuracy {best_run['combined_metric_accuracy']}")
            print(f"Best model hyperparameters: {best_run['config']}")

            model_filename = 'best_model_NN_v1.pth'
            json_filename = 'best_model_details.json'
            entire_model_filename = 'best_model_NN_v1_entire.pth'

            # Check for the existence of the previous best model and compare performance
            if os.path.exists(model_filename):
                try:
                    previous_best = torch.load(model_filename)
                    if previous_best['combined_metric_loss'] <= best_run['combined_metric_loss']:
                        print("Previous best model is better or equal. Not overwriting.")
                    else:
                        print("New model is better. Overwriting previous best model.")
                        save_model_and_details(best_run, model_filename, json_filename, entire_model_filename, model, best_scaler_path)
                except Exception as e:
                    print(f"Error loading previous best model: {e}. Saving new model.")
                    save_model_and_details(best_run, model_filename, json_filename, entire_model_filename, model, best_scaler_path)
            else:
                save_model_and_details(best_run, model_filename, json_filename, entire_model_filename, model, best_scaler_path)

            print(f"Best model saved as {model_filename}")
        else:
            print("No valid runs found with non-NaN metrics.")
    else:
        print("No valid run results found.")

    

    # Cleanup temporary scaler files at the end of the script
    print("Cleaning up temporary normalization files...")
    shutil.rmtree(temp_dir)
    print(f"Temporary normalization directory '{temp_dir}' has been deleted.")