function python_dask_run_HPPC_interpolation_iterations(start_idx, end_idx, profile_name, training_set_ratio)
    %% Set up the paths and directories

    % Specify the paths in the NFS root directory
    path_nfs_root = fullfile('\\wsl$\Ubuntu-22.04\mnt\shared');
    path_simulink_data_gen = fullfile(path_nfs_root, 'simulink_data_generator');
    path_base_battery_model_params = fullfile(path_simulink_data_gen, 'base_battery_model_params');
    path_HPPC_base_params = fullfile(path_simulink_data_gen, 'HPPC_base_params');
    path_HPPC_interpolated_params = fullfile(path_simulink_data_gen, 'HPPC_interpolated_params');
    path_temp_slices = fullfile(path_HPPC_interpolated_params, 'temp_slices');    

    % Load the HPPC data
    load(fullfile(path_HPPC_base_params, 'SOC_HPPC.mat'), 'SOC_HPPC');
    load(fullfile(path_HPPC_base_params, 'R1_HPPC.mat'), 'R1_HPPC');
    load(fullfile(path_HPPC_base_params, 'R2_HPPC.mat'), 'R2_HPPC');
    load(fullfile(path_HPPC_base_params, 'C1_HPPC.mat'), 'C1_HPPC');
    load(fullfile(path_HPPC_base_params, 'C2_HPPC.mat'), 'C2_HPPC');

    % Load the SOC vector from the simulink base battery model
    load(fullfile(path_base_battery_model_params, 'SOC_vec_sim.mat'), 'SOC_vec_sim');

    % Initialize local temporary storage for this slice
    slice_size = end_idx - start_idx + 1;

    % Preallocate storage for optimized parameters and errors
    local_best_params = struct('R1', [], 'R2', [], 'C1', [], 'C2', []);
    local_best_errors = struct('R1', inf, 'R2', inf, 'C1', inf, 'C2', inf);

    % Initialize temporary storage for parallel loop
    temp_best_params = repmat({local_best_params}, slice_size, 1);
    temp_best_errors = repmat({local_best_errors}, slice_size, 1);

    % Start or refresh the parallel pool
    refresh_parallel_pool(profile_name);

    % Perform the assigned iterations in a parfor loop
    parfor iter = 1:slice_size      

        % Randomly select indices from the SOC_HPPC data for training and validation
        rand_indices = randperm(length(SOC_HPPC));
        training_size = round(training_set_ratio * length(SOC_HPPC));
        training_indices = rand_indices(1:training_size);
        validation_indices = rand_indices(training_size + 1:end);

        % Update the training and validation sets
        training_SOC = SOC_HPPC(training_indices);
        validation_SOC = SOC_HPPC(validation_indices);

        % Parameters for training
        training_R1 = R1_HPPC(training_indices);
        training_R2 = R2_HPPC(training_indices);
        training_C1 = C1_HPPC(training_indices);
        training_C2 = C2_HPPC(training_indices);

        % Parameters for validation
        validation_R1 = R1_HPPC(validation_indices);
        validation_R2 = R2_HPPC(validation_indices);
        validation_C1 = C1_HPPC(validation_indices);
        validation_C2 = C2_HPPC(validation_indices);

        % Perform the interpolation for each parameter
        interp_R1 = interp1(training_SOC, training_R1, validation_SOC, 'pchip');
        interp_R2 = interp1(training_SOC, training_R2, validation_SOC, 'pchip');
        interp_C1 = interp1(training_SOC, training_C1, validation_SOC, 'pchip');
        interp_C2 = interp1(training_SOC, training_C2, validation_SOC, 'pchip');

        % Calculate the errors for each parameter
        error_R1 = mean(abs(validation_R1 - interp_R1));
        error_R2 = mean(abs(validation_R2 - interp_R2));
        error_C1 = mean(abs(validation_C1 - interp_C1));
        error_C2 = mean(abs(validation_C2 - interp_C2));

        % Store local results in temporary variables
        local_errors = struct('R1', error_R1, 'R2', error_R2, 'C1', error_C1, 'C2', error_C2);
        local_params = struct('R1', interp1(training_SOC, training_R1, SOC_vec_sim, 'pchip'), ...
                              'R2', interp1(training_SOC, training_R2, SOC_vec_sim, 'pchip'), ...
                              'C1', interp1(training_SOC, training_C1, SOC_vec_sim, 'pchip'), ...
                              'C2', interp1(training_SOC, training_C2, SOC_vec_sim, 'pchip'));

        % Update the temporary storage
        temp_best_params{iter} = local_params;
        temp_best_errors{iter} = local_errors;        
    end

    % Save the results for this slice to a MAT file using the start and end indices in the filename
    save(fullfile(path_temp_slices, sprintf('python_dask_HPPC_interpolation_results_%d_%d.mat', start_idx, end_idx)), ...
        'temp_best_params', 'temp_best_errors', '-v7.3');  % Specify MAT-file version 7.3 for larger variables  
end


%% Function to start or refresh parallel pool
function refresh_parallel_pool(profile_name)
    % Check if a parallel pool is already running
    poolobj = gcp('nocreate');
    if ~isempty(poolobj)
        delete(poolobj); % Delete existing pool
    end
    % Start a new parallel pool with the specified profile
    parpool(profile_name);
end
