%% SPEED RAMP TEST AT CONSTANT STEER
% Change Conditions in input to get wanted test: in Simulink
d0 = 20; % Steering costant value
IN_PED = 0.5; % Pedal costant value

V0 = 50/3.6; % Initial speed
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
elapsed_time_simulation = toc;
fprintf('Simulation completed\n')
fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)

%% Post-Processing
dataAnalysis(model_sim,vehicle_data,Ts);
%% Normalised axle characteristics
%% Handling diagram
%% Compare the theoretical with the fitted steering gradients.
%% Effect of suspension roll stiffness, camber and toe angle effect on steering characteristics