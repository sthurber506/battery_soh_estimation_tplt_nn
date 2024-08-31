function python_dask_save_interpolated_parameters()
    %% Set up the paths and directories

    % Specify the paths in the NFS root directory
    path_nfs_root = fullfile('\\wsl$\Ubuntu-22.04\mnt\shared');
    path_simulink_data_gen = fullfile(path_nfs_root, 'simulink_data_generator');
    path_base_battery_model_params = fullfile(path_simulink_data_gen, 'base_battery_model_params');
    path_HPPC_base_params = fullfile(path_simulink_data_gen, 'HPPC_base_params');
    path_HPPC_interpolated_params = fullfile(path_simulink_data_gen, 'HPPC_interpolated_params');
    path_temp_slices = fullfile(path_HPPC_interpolated_params, 'temp_slices');
    path_simulink_model = fullfile(path_simulink_data_gen, 'simulink_model');
    path_simulink_results = fullfile(path_simulink_data_gen, 'simulink_results');

    
    % Load the best parameters
    load(fullfile(path_HPPC_interpolated_params, 'python_dask_best_HPPC_interpolation_results.mat'), 'best_params');
    
    % Save each parameter as a separate .mat file
    R1_HPPC_interpolated = best_params.R1;
    R2_HPPC_interpolated = best_params.R2;
    C1_HPPC_interpolated = best_params.C1;
    C2_HPPC_interpolated = best_params.C2;
    
    save(fullfile(path_HPPC_interpolated_params, 'R1_HPPC_interpolated.mat'), 'R1_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'R2_HPPC_interpolated.mat'), 'R2_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'C1_HPPC_interpolated.mat'), 'C1_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'C2_HPPC_interpolated.mat'), 'C2_HPPC_interpolated');
    
    % Load all the .mat files from the base battery model parameters folder
    files = dir(fullfile(path_base_battery_model_params, '*.mat'));
    for file = files'
        load(fullfile(file.folder, file.name));
    end

    % Get the number of temperatures
    numTemps = length(T_vec_sim);

    % Calculate τ₁ and τ₂ from the interpolated parameters
    tau1_HPPC_interpolated = best_params.R1 .* best_params.C1;
    tau2_HPPC_interpolated = best_params.R2 .* best_params.C2;

    % Replicate the column vectors to match the number of temperature points
    R1_mat_HPPC_interpolated = repmat(best_params.R1, 1, numTemps);
    R2_mat_HPPC_interpolated = repmat(best_params.R2, 1, numTemps);
    tau1_mat_HPPC_interpolated = repmat(tau1_HPPC_interpolated, 1, numTemps);
    tau2_mat_HPPC_interpolated = repmat(tau2_HPPC_interpolated, 1, numTemps);

    % Save the calculated matrices
    save(fullfile(path_HPPC_interpolated_params, 'tau1_HPPC_interpolated.mat'), 'tau1_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'tau2_HPPC_interpolated.mat'), 'tau2_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'R1_mat_HPPC_interpolated.mat'), 'R1_mat_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'R2_mat_HPPC_interpolated.mat'), 'R2_mat_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'tau1_mat_HPPC_interpolated.mat'), 'tau1_mat_HPPC_interpolated');
    save(fullfile(path_HPPC_interpolated_params, 'tau2_mat_HPPC_interpolated.mat'), 'tau2_mat_HPPC_interpolated');
    
    % Print parameter values and errors
    disp('Best Parameters:');
    disp(best_params);

















    % Load the Simulink model
    model = 'two_pulse_test_for_vmax_for_python';
    load_system(fullfile(path_simulink_model, model));

    % Block to set parameters
    block = [model '/battery_mod_1'];

    % Set the original parameters
    set_param(block, 'SOC_vec', mat2str(SOC_vec_sim));
    set_param(block, 'T_vec', mat2str(T_vec_sim));
    set_param(block, 'T_vec_unit', 'Celsius');
    set_param(block, 'V0_mat', mat2str(OCV_mat_sim));
    set_param(block, 'R0_mat', mat2str(R0_mat_sim));
    set_param(block, 'AH', mat2str(AHC_nom));
    set_param(block, 'N0vec', mat2str(Ncycles_vec_sim));
    set_param(block, 'dV0vec', mat2str(dV0_N_vec_sim));
    set_param(block, 'dAHvec', mat2str(dVAHC_N_vec_sim));

    % Set the 'prm_dyn' parameter to 'rc2' for two time-constant dynamics
    set_param(block, 'prm_dyn', 'simscape.enum.tablebattery.prm_dyn.rc2');

    % Convert matrices to strings for setting Simulink block parameters
    R1_mat_HPPC_interpolated_str = mat2str(R1_mat_HPPC_interpolated);
    R2_mat_HPPC_interpolated_str = mat2str(R2_mat_HPPC_interpolated);
    tau1_mat_HPPC_interpolated_str = mat2str(tau1_mat_HPPC_interpolated);
    tau2_mat_HPPC_interpolated_str = mat2str(tau2_mat_HPPC_interpolated);

    % Set the replicated matrices as parameters in the battery block
    set_param(block, 'R1_mat', R1_mat_HPPC_interpolated_str);
    set_param(block, 'R2_mat', R2_mat_HPPC_interpolated_str);
    set_param(block, 'tau1_mat', tau1_mat_HPPC_interpolated_str);
    set_param(block, 'tau2_mat', tau2_mat_HPPC_interpolated_str);

    % Set fade characteristic vectors to zeros
    fade_char_zero_vector_str = mat2str(zeros(1, length(Ncycles_vec_sim)));
    set_param(block, 'dR1vec', fade_char_zero_vector_str);
    set_param(block, 'dR2vec', fade_char_zero_vector_str);

    % Save the updated Simulink model
    save_system(model);
end
