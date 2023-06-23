%% Data Preprocessing
% Function to clean data before analysis
% ---------------------------------------------------------
function model_sim_cut = data_prep(model_sim,t_remove)

    % t_remove [s] time span we want to cut from the beginning

    %% Extract data from simulink model
    % ---------------------------------
    time_sim = model_sim.states.u.time;
    dt = time_sim(2)-time_sim(1); % [s] Simulation time step

    
    nsample_remove = t_remove/dt; % number of samples to cut
%     model_sim_cut = model_sim(nsample_remove:end);

end