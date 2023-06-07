%% SPEED RAMP TEST AT CONSTANT STEER
%% Initialization
% ----------------------------
clc
clear
close all

initialize_environment;

% ----------------------------
%% Load vehicle data
% ----------------------------

% test_tyre_model; % some plot to visualize the curvers resulting from the
% loaded data

vehicle_data = getVehicleDataStruct();
% pacejkaParam = loadPacejkaParam();

%% Testing conditions
% Change Conditions in input to get wanted test: in Simulink
test_type = 'speed_ramp';
slope = 0.1; % slope of ramp
d0 = 20; % Steering angle initial/costant value
u0 = 40; % Forward speed initial/constant value
IN_PED = 0.1; % Pedal costant value

V0 = u0/3.6; % Initial speed
X0 = loadInitialConditions(V0); % Initial forward speed

%% Simulation parameters
simulationPars = getSimulationParams(); 
Ts = simulationPars.times.step_size;  % integration step for the simulation (fixed step)
T0 = simulationPars.times.t0;         % starting time of the simulation
Tf = simulationPars.times.tf;         % stop time of the simulation

%% Start Simulation
fprintf('Starting Simulation\n')
tic;
model_sim = sim('SteerTestsSS.slx');
% model_sim = sim('SteerTestsSS_CONTROL.slx');
elapsed_time_simulation = toc;
fprintf('Simulation completed\n')
fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)

%% Post-Processing
dataAnalysis(model_sim,vehicle_data,Ts);

%% Normalised axle characteristics
% They are computed in simulink and stored under "extra_params"
axle_character(model_sim,vehicle_data,Ts)

%% Handling diagram

%% Compare the theoretical with the fitted steering gradients.

%% Effect of suspension roll stiffness, camber and toe angle effect on steering characteristics
% Twitch the characteristic angles to observe how the behaviour changes