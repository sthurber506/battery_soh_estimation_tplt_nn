function python_dask_two_pulse_test_for_vmax(model_path, sim_time, start_idx, end_idx, pulse_start, pulse_duration, profile_name, AHC_nom)
    % Define the NFS root path
    path_nfs_root = fullfile('\\wsl$\Ubuntu-22.04\mnt\shared');
    path_simulink_data_gen = fullfile(path_nfs_root, 'simulink_data_generator');
    path_simulink_results = fullfile(path_simulink_data_gen, 'simulink_results');    
    path_base_battery_model_params = fullfile(path_simulink_data_gen, 'base_battery_model_params');

    % Load the SOC_and_OCV_table from the .mat file
    load(fullfile(path_base_battery_model_params, 'SOC_and_OCV_table.mat'), 'SOC_and_OCV_table');    
    
    % Check for local user directories and set the local Simulink model path
    if isfolder('C:\Users\Shaun_Thurber\Documents')
        local_simulink_model_path = 'C:\Users\Shaun_Thurber\Documents\SimulinkModels\two_pulse_test_for_vmax_for_python.slx';
        ip_address_path = 'C:\Users\Shaun_Thurber\Documents\SimulinkModels\ip_address.txt';
    elseif isfolder('C:\Users\sthur\Downloads')
        local_simulink_model_path = 'C:\Users\sthur\Downloads\SimulinkModels\two_pulse_test_for_vmax_for_python.slx';
        ip_address_path = 'C:\Users\sthur\Downloads\SimulinkModels\ip_address.txt';
    else
        error('No valid user directory found.');
    end

    % Read the IP address from the text file
    ip_address = strtrim(fileread(ip_address_path));

    % Debugging output to confirm file paths
    disp(['Simulink results path: ', path_simulink_results]);
    disp(['Local Simulink model path: ', local_simulink_model_path]);
    disp(['IP address: ', ip_address]);

    % Check if the file exists before loading the model
    if exist(local_simulink_model_path, 'file') ~= 2
        error('The Simulink model file does not exist at the specified path.');
    end

    % Load the model
    load_system(local_simulink_model_path);
    
    % Extract the model name from the file path
    [~, model_name, ~] = fileparts(local_simulink_model_path);

    % Load the worker simulation ranges CSV file
    worker_simulation_ranges = readtable(fullfile(path_simulink_data_gen, 'worker_simulation_ranges.csv'));

    % Extract the relevant slice for this worker based on IP address
    simulations_slice = worker_simulation_ranges(strcmp(worker_simulation_ranges.worker_ip, ip_address), :);
    fprintf('Number of simulations retrieved: %d\n', height(simulations_slice));

    % Check existing results and remove corresponding rows from simulations_slice
    simulations_slice = remove_completed_simulations(path_simulink_results, simulations_slice);
    fprintf('Number of simulations to be processed after checking existing results: %d\n', height(simulations_slice));
    

    % Update the total_simulations calculation to use the simulations_slice
    total_simulations = height(simulations_slice);
    simIn(total_simulations) = Simulink.SimulationInput(model_name);

    disp('Simulation Parameters:');
    disp(['Model Path: ', model_path]);
    disp(['Simulation Time: ', num2str(sim_time)]);
    disp(['Pulse Start: ', num2str(pulse_start)]);
    disp(['Pulse Duration: ', num2str(pulse_duration)]);
    disp(['Profile Name: ', profile_name]);
    disp(['AHC Nom: ', num2str(AHC_nom)]);
    disp('--------------------------');
    disp(['Total Simulations: ', num2str(total_simulations)]);
    
    % Define batch size based on IP address
    batch_size = get_batch_size_based_on_ip(ip_address);
    num_batches = ceil(total_simulations / batch_size);

    % Start or refresh the parallel pool
    refresh_parallel_pool(profile_name);

    % Loop through each batch
    for batch_num = 1:num_batches
        % Determine the start and end indices for this batch
        batch_start = (batch_num - 1) * batch_size + 1;
        batch_end = min(batch_num * batch_size, total_simulations);
        current_batch_size = batch_end - batch_start + 1;

        % Configure each SimulationInput object for the current batch
        parfor idx = batch_start:batch_end
            SOC_val = simulations_slice.SOC(idx);
            capacity_available_val = simulations_slice.capacity(idx);
            c_rate_val = simulations_slice.c_rate(idx);

            % Create time vector and initialize current profile
            dt = 0.001; % Time step in seconds
            t = 0:dt:str2double(sim_time); 
                
            % Calculate pulse current for current C_rate
            pulse_current_val = -c_rate_val * AHC_nom;
            
            % Define the pulse current profile for the current simulation
            I_pulse = zeros(size(t));
            
            % Applying the pulses to the current profile
            for pulse = 1:length(pulse_start)
                pulse_indices = t >= pulse_start(pulse) & t < pulse_start(pulse) + pulse_duration;
                I_pulse(pulse_indices) = pulse_current_val;
            end
            
            % Combine the time and current vectors into a two-column matrix
            load_profile_i = [t', I_pulse'];
            
            % Set up simulation input
            simIn(idx) = Simulink.SimulationInput(model_name);
            simIn(idx) = simIn(idx).setVariable('load_profile', load_profile_i);
            simIn(idx) = simIn(idx).setBlockParameter([model_name '/battery_mod_1'], 'stateOfCharge', num2str(SOC_val));
            simIn(idx) = simIn(idx).setBlockParameter([model_name '/battery_mod_1'], 'AH', num2str(capacity_available_val));
            simIn(idx) = simIn(idx).setModelParameter('StopTime', num2str(sim_time));
            simIn(idx) = simIn(idx).setVariable('dt', dt);

            % Inside the parfor loop, after setting up the simulation input
            fprintf('Simulation %d: SOC = %.2f, Capacity = %.2f, C-rate = %.2f\n', idx, SOC_val, capacity_available_val, c_rate_val);
        end
        
        % Options for parsim
        options = struct;
        options.UseFastRestart = 'off';
        options.ShowProgress = 'on';
        options.TransferBaseWorkspaceVariables = 'on';
        
        % Run parallel simulations for the current batch with options
        simOut = parsim(simIn(batch_start:batch_end), options);

        % After running parsim, before processing results
        disp(['Batch ', num2str(batch_num), ' of ', num2str(num_batches), ' completed.']);
        
        % Process the results of the parallel simulations for the current batch
        for idx = batch_start:batch_end
            if ~isempty(simOut(idx - batch_start + 1).ErrorMessage)
                fprintf('Error in simulation %d: %s\n', idx, simOut(idx - batch_start + 1).ErrorMessage);
            else
                % Extract the simulation results
                v_term_pulse_sim = simOut(idx - batch_start + 1).get('v_term');
                soc_pulse_sim = simOut(idx - batch_start + 1).get('SOC');
                temperature_pulse_sim = simOut(idx - batch_start + 1).get('temperature');
                load_pulse_sim = simOut(idx - batch_start + 1).get('load');

                SOC_val = simulations_slice.SOC(idx);
                capacity_available_val = simulations_slice.capacity(idx);
                c_rate_val = simulations_slice.c_rate(idx);

                % Extract data from timeseries objects
                v_term_pulse_sim_data = v_term_pulse_sim.Data;
                soc_pulse_sim_data = soc_pulse_sim.Data;
                temperature_pulse_sim_data = temperature_pulse_sim.Data;
                load_pulse_sim_data = load_pulse_sim.Data;

                % Ensure time is extracted correctly from one of the timeseries
                t = v_term_pulse_sim.Time;

                simulation_table = table(t, v_term_pulse_sim_data, load_pulse_sim_data, temperature_pulse_sim_data, soc_pulse_sim_data, ...
                    'VariableNames', {'Time', 'Voltage', 'Current', 'Temperature', 'SOC'});
                simulation_table.SOC_initialized = repmat(SOC_val, size(simulation_table, 1), 1);
                simulation_table.Capacity_initialized = repmat(capacity_available_val, size(simulation_table, 1), 1);
                simulation_table.C_rate_initialized = repmat(c_rate_val, size(simulation_table, 1), 1);

                % Add another column for soc_pulse_sim_data that is rounded to 2 decimal places
                simulation_table.SOC_rounded = round(simulation_table.SOC, 2);

                % Add another column for the OCV value that corresponds to the rounded SOC value by looking up the SOC_and_OCV_table
                % Find the indices of the rounded SOC values in the SOC_and_OCV_table
                [~, idx_ocv] = ismember(simulation_table.SOC_rounded, SOC_and_OCV_table.SOC_vec_sim);

                % Use these indices to map the OCV values to the simulation_table
                simulation_table.OCV = SOC_and_OCV_table.V0_25C(idx_ocv);

                simulation_struct = table2struct(simulation_table, 'ToScalar', true);

                % Generate a file name based on the SOC, capacity, and C-rate values
                file_name = sprintf('AHC_%.2f_Crate_%.2f_SOC_%.2f.mat', capacity_available_val, c_rate_val, SOC_val);
                file_path = fullfile(path_simulink_results, file_name);

                try
                    save(file_path, '-struct', 'simulation_struct');
                catch ME
                    fprintf('Error saving simulation %d: %s\n', idx, ME.message);
                end            
            end
        end

        % Clear variables to free up memory before the next batch
        clear simIn simOut;
        disp(['Batch ', num2str(batch_num), ' processed and saved. Memory cleared.']);
    end

    % Clean up parallel pool
    delete(gcp('nocreate'));
end

% Function to start or refresh parallel pool
function refresh_parallel_pool(profile_name)
    % Check if a parallel pool is already running
    poolobj = gcp('nocreate');
    if ~isempty(poolobj)
        delete(poolobj); % Delete existing pool
    end
    % Start a new parallel pool with the specified profile
    parpool(profile_name);
end

% Function to get batch size based on IP address
function batch_size = get_batch_size_based_on_ip(ip_address)
    % Define a mapping of IP addresses to batch sizes
    batch_size_map = containers.Map;
    batch_size_map('100.82.123.17') = 120; % URIPEL
    batch_size_map('100.73.226.71') = 30; % FCAE430Tower
    batch_size_map('100.65.124.63') = 20; % DESKTOP-C1HPDQR 
    batch_size_map('100.72.43.78') = 20; % DESKTOP-AMQ1U85
    batch_size_map('100.113.204.47') = 40; % DESKTOP-69BT84J ~ Maingear Laptop
    batch_size_map('100.75.156.93') = 8; % DESKTOP-LO3754S ~ Home Desktop
    batch_size_map('100.82.151.63') = 8; % DESKTOP-8I71MUM ~ Dell Optiplex Left 
    batch_size_map('100.120.1.68') = 12; % DESKTOP-PUG0INT ~ Dell Optiplex Middle
    batch_size_map('100.123.66.46') = 4; % DESKTOP-76MRIAP ~ Dell Optiplex Right
    
    % Default batch size if IP address is not found
    default_batch_size = 8;
    
    % Get batch size based on IP address
    if isKey(batch_size_map, ip_address)
        batch_size = batch_size_map(ip_address);
    else
        batch_size = default_batch_size;
        disp('DEFAULT BATCH SIZE USED ===============================================================================================================');
    end
end


% Function to remove completed simulations
function simulations_slice = remove_completed_simulations(path_simulink_results, simulations_slice)
    % Initialize a logical array to mark completed simulations
    completed_simulations = false(height(simulations_slice), 1);

    % Loop through each simulation slice
    for idx = 1:height(simulations_slice)
        SOC_val = simulations_slice.SOC(idx);
        capacity_available_val = simulations_slice.capacity(idx);
        c_rate_val = simulations_slice.c_rate(idx);

        % Generate the expected file name for this simulation
        file_name = sprintf('AHC_%.2f_Crate_%.2f_SOC_%.2f.mat', capacity_available_val, c_rate_val, SOC_val);
        file_path = fullfile(path_simulink_results, file_name);

        % Check if the file exists
        if exist(file_path, 'file') == 2
            completed_simulations(idx) = true;
        end
    end

    % Remove completed simulations from the slice
    simulations_slice(completed_simulations, :) = [];
end