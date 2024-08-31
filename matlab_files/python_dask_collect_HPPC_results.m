function python_dask_collect_HPPC_results()
    %% Set up the paths and directories

    % Specify the paths in the NFS root directory
    path_nfs_root = fullfile('\\wsl$\Ubuntu-22.04\mnt\shared');
    path_simulink_data_gen = fullfile(path_nfs_root, 'simulink_data_generator');
    path_base_battery_model_params = fullfile(path_simulink_data_gen, 'base_battery_model_params');
    path_HPPC_base_params = fullfile(path_simulink_data_gen, 'HPPC_base_params');
    path_HPPC_interpolated_params = fullfile(path_simulink_data_gen, 'HPPC_interpolated_params');
    path_temp_slices = fullfile(path_HPPC_interpolated_params, 'temp_slices');    

    % Preallocate storage for optimized parameters and errors
    best_params = struct('R1', [], 'R2', [], 'C1', [], 'C2', []);
    best_errors = struct('R1', inf, 'R2', inf, 'C1', inf, 'C2', inf);

    % Loop through all result files and aggregate the best results
    files = dir(fullfile(path_temp_slices, 'python_dask_HPPC_interpolation_results_*.mat'));
    for file = files'
        load(fullfile(file.folder, file.name), 'temp_best_params', 'temp_best_errors');
        
        for iter = 1:length(temp_best_params)
            localErrors = temp_best_errors{iter};
            localParams = temp_best_params{iter};
            
            if localErrors.R1 < best_errors.R1
                best_errors.R1 = localErrors.R1;
                best_params.R1 = localParams.R1;
            end
            if localErrors.R2 < best_errors.R2
                best_errors.R2 = localErrors.R2;
                best_params.R2 = localParams.R2;
            end
            if localErrors.C1 < best_errors.C1
                best_errors.C1 = localErrors.C1;
                best_params.C1 = localParams.C1;
            end
            if localErrors.C2 < best_errors.C2
                best_errors.C2 = localErrors.C2;
                best_params.C2 = localParams.C2;
            end
        end
    end

    % Save the best parameters and errors
    save(fullfile(path_HPPC_interpolated_params, 'python_dask_best_HPPC_interpolation_results.mat'), 'best_params', 'best_errors');

    % Plotting Best Interpolations Against True HPPC Values
    % Load the HPPC data
    load(fullfile(path_HPPC_base_params, 'SOC_HPPC.mat'), 'SOC_HPPC');
    load(fullfile(path_HPPC_base_params, 'R1_HPPC.mat'), 'R1_HPPC');
    load(fullfile(path_HPPC_base_params, 'R2_HPPC.mat'), 'R2_HPPC');
    load(fullfile(path_HPPC_base_params, 'C1_HPPC.mat'), 'C1_HPPC');
    load(fullfile(path_HPPC_base_params, 'C2_HPPC.mat'), 'C2_HPPC');

    % Load the SOC vector from the simulink base battery model
    load(fullfile(path_base_battery_model_params, 'SOC_vec_sim.mat'), 'SOC_vec_sim');

    % Function to setup the plotting environment
    setupPlottingEnvironment_v2(1, true);
    params = {'R1', 'R2', 'C1', 'C2'};
    titles = {'R1 Interpolation with PCHIP', 'R2 Interpolation with PCHIP', 'C1 Interpolation with PCHIP', 'C2 Interpolation with PCHIP'};
    yLabels = {'R1 (Ohms)', 'R2 (Ohms)', 'C1 (F)', 'C2 (F)'};
    
    for i = 1:4
        subplot(4,1,i);
        plot(SOC_HPPC, eval([params{i}, '_HPPC']), 'o', SOC_vec_sim, best_params.(params{i}), '-');
        title(titles{i});
        xlabel('SOC (%)');
        ylabel(yLabels{i});
        legend('HPPC Data', 'Interpolated Data', 'Location', 'best');
        set(gca, 'FontSize', 18);
    end
    
    % Save the plot as a PNG image
    saveas(gcf, fullfile(path_HPPC_interpolated_params, 'HPPC_interpolation_comparison.png'));

    % Print the best parameters and their corresponding errors
    disp('Best Parameters and Corresponding Errors:');
    for i = 1:4
        fprintf('%s: Best Param = %s, Error = %f\n', params{i}, mat2str(best_params.(params{i})), best_errors.(params{i}));
    end
end

function setupPlottingEnvironment_v2(figureNumber, maximize)
    % This function sets up the plotting environment according to specified standards.
    
    % Check if the figure already exists; if not, create a new one.
    fig = figure(figureNumber);
    clf(fig);  % Clear the figure to start fresh
    
    % Set the font size for all text in the figure
    set(fig, 'DefaultAxesFontSize', 18);
    set(fig, 'DefaultTextFontSize', 18);
    
    if maximize == true
        % Move the figure window to the primary monitor and maximize it
        set(fig, 'Units', 'normalized', 'OuterPosition', [0, 0, 1, 1]);
        
        % Ensure the figure window is drawn and maximized before plotting
        drawnow;
        pause(0.5);  % Pause briefly to ensure the figure window is ready
    
        % Bring the figure to the front
        figure(fig);    
        
        % When ready, display the figure window
        movegui(gcf, 'center');  % Center on screen if needed
        set(gcf, 'WindowState', 'maximized');  % Maximize the figure window
    end
end