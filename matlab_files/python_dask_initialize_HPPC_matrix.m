close all;
clear;
clc;

%% Set up the paths and directories

% Specify the paths in the NFS root directory
path_nfs_root = fullfile('\\wsl$\Ubuntu-22.04\mnt\shared');
path_simulink_data_gen = fullfile(path_nfs_root, 'simulink_data_generator');
path_base_battery_model_params = fullfile(path_simulink_data_gen, 'base_battery_model_params');
path_HPPC_base_params = fullfile(path_simulink_data_gen, 'HPPC_base_params');
path_HPPC_interpolated_params = fullfile(path_simulink_data_gen, 'HPPC_interpolated_params');
path_temp_slices = fullfile(path_HPPC_interpolated_params, 'temp_slices');

% Read max_iterations value from text file
fileID = fopen(fullfile(path_HPPC_interpolated_params, 'max_iterations.txt'), 'r');
max_iterations = fscanf(fileID, '%d');
fclose(fileID);

% Initialize data structures for parameters and errors
error_thresholds = struct('R1', 0.01, 'R2', 0.01, 'C1', 0.01, 'C2', 0.01);
best_params = struct('R1', [], 'R2', [], 'C1', [], 'C2', []);
best_errors = struct('R1', inf, 'R2', inf, 'C1', inf, 'C2', inf);

% Preallocate storage for parameters and errors
temp_best_params = repmat({best_params}, max_iterations, 1);
temp_best_errors = repmat({best_errors}, max_iterations, 1);

% Save the initialized variables to a MAT file
save(fullfile(path_HPPC_interpolated_params, 'python_dask_initialized_HPPC_interpolation_matrix.mat'), 'temp_best_params', 'temp_best_errors', 'max_iterations');