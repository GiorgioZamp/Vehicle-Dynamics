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
    flg = 1; % flag for simulation profile selection
    
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
    dataAnalysis(model_sim,vehicle_data,Ts);
    handling_diagram(model_sim,vehicle_data);
    % vehicleAnimation(model_sim,vehicle_data,Ts); % needs Clothoids Toolbox

% ----------------------------
%% Camber, Toe Angle and Roll Stiffness Effects
   
    % Camber Effect
    camber_set = -10:2:+10; % [deg] camber angle
    camber_effect(camber_set,vehicle_data)

    % Toe Effect
    toe_set = -3:1:+3; % [deg] toe angle
    toe_effect(toe_set,vehicle_data)

    % Roll Stiffness Effect
	% Vary Roll Stiffnesses ratio
	% e_phi->1 all to the front
	% e_phi->0 all to the back
    e_phi_set = [0.2,0.5,0.8];
    rollstiff_effect(e_phi_set,vehicle_data)

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

%     handling_diagram(model_sim,vehicle_data,Ts); % MY FUNCTION

%     vehicleAnimation(model_sim,vehicle_data,Ts);

%     % ----------------------------