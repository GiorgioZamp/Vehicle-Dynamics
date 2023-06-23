%% DOUBLE TRACK VEHICLE MODEL
% Main script for a basic simulation framework with a double track vehcile model
%  authors: 
%  rev. 1.0 Mattia Piccinini & Gastone Pietro Papini Rosati
%  rev. 2.0 Edoardo Pagot
%  date:
%  rev 1.0:    13/10/2020
%  rev 2.0:    16/05/2022
%  rev 2.1:    08/07/2022 (Biral)
%       - added Fz saturation. Correceted error in Fx
%       - initial condition is now parametric in initial speed
%       - changed the braking torque parameters to adapt to a GP2 model
% ----------------------------------------------------------------

% ----------------------------
%% Initialization
% ----------------------------
clc
close all
clear
initialize_environment;
% ----------------------------
%% Load vehicle data
% ----------------------------

% test_tyre_model; % some plot to visualize the curvers resulting from the
% loaded data

vehicle_data = getVehicleDataStruct();
% pacejkaParam = loadPacejkaParam();

% ----------------------------
%% SPEED RAMP TEST
    % ----------------------------
    global flg;
    flg = 1; % flag for simulation profile selection
    
    % P=0.0436462066049939 I=0.0175035310088279 D=-0.00909870879334459 N=2
    % ----------------------------
    % Define initial conditions for the simulation
    % ----------------------------
    V0 = 50/3.6;        % [m/s] Initial speed
    X0 = loadInitialConditions(V0);
    steer_des = 10;     % [°] desired steer angle
    % ----------------------------
    % Define the desired speed
    % ----------------------------
    V_des = 50/3.6; % Desired speed for controller (kept for ease)
    slope = 0.5;    % Slope of speed ramp +1 m/s each 10 s [m/s^2]

    % ----------------------------
    % Simulation parameters
    % ----------------------------
    simulationPars = getSimulationParams(); 
    Ts = simulationPars.times.step_size;  % integration step for the simulation (fixed step)
    T0 = simulationPars.times.t0;         % starting time of the simulation
    Tf = simulationPars.times.tf;         % stop time of the simulation
    
    % ----------------------------
    % Start Simulation
    % ----------------------------
    fprintf('Starting Simulation\n')
    tic;
    model_sim = sim('Vehicle_Model_2Track_OLD');
    elapsed_time_simulation = toc;
    fprintf('Simulation completed\n')
    fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)
    
    % ----------------------------
    % Post-Processing
    % ----------------------------
%     t_remove = 0.4; % cutting time frame
%     model_sim = data_prep(model_sim,t_remove);
    dataAnalysis(model_sim,vehicle_data,Ts);
%     vehicleAnimation(model_sim,vehicle_data,Ts); %Clothoids Toolbox not
%     working
    handling_diagram(model_sim,vehicle_data,Ts); % MY FUNCTION
    % ----------------------------

%% STEER RAMP TEST
% ----------------------------
%       flg = 2; % flag for simulation profile selection

%     % Define initial conditions for the simulation
%     % ----------------------------
%     V0 = 50/3.6; % Initial speed
%     X0 = loadInitialConditions(V0);
%     
%     
%     % ----------------------------
%     % Define the desired steer
%     % ----------------------------
%     V_des = 50/3.6; % Desired speed for controller
%     steer_des = 10;     % [°] desired steer angle
%     slope = 0.1;    % Slope of steer ramp
%     
%     % ----------------------------
%     % Simulation parameters
%     % ----------------------------
%     simulationPars = getSimulationParams(); 
%     Ts = simulationPars.times.step_size;  % integration step for the simulation (fixed step)
%     T0 = simulationPars.times.t0;         % starting time of the simulation
%     Tf = simulationPars.times.tf;         % stop time of the simulation
%     
%     % ----------------------------
%     % Start Simulation
%     % ----------------------------
%     fprintf('Starting Simulation\n')
%     tic;
%     model_sim = sim('Vehicle_Model_2Track_OLD');
%     elapsed_time_simulation = toc;
%     fprintf('Simulation completed\n')
%     fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)
%     
%     % ----------------------------
%     % Post-Processing
%     % ----------------------------
%     dataAnalysis(model_sim,vehicle_data,Ts);
%     vehicleAnimation(model_sim,vehicle_data,Ts);
%     % ----------------------------