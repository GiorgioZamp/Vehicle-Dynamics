% -----------------------
%% Initialization
% -----------------------

clc
clearvars 
close all   
format long;

% Set LaTeX as default interpreter for axis labels, ticks and legends
set(0,'defaulttextinterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesXGrid','on')
set(groot,'defaultAxesYGrid','on')

set(0,'DefaultFigureWindowStyle','docked');
set(0,'defaultAxesFontSize',18)
set(0,'DefaultLegendFontSize',18)

addpath(genpath('Lib_Vehicle'));
addpath(genpath('Longit_Controller'));
addpath(genpath('Utilities'));
addpath(genpath('../tyre_model/tyre_lib'));
addpath(genpath('submodules'));
addpath(genpath('Graphs'));

% --------------------
%% Open Simulink model
% --------------------
% Check if the Simulink model is already opened. Otherwise open it (_OLD
% for compatibility with older matlab versions)
openModels = find_system('SearchDepth', 0);
if (isempty(find(strcmp(openModels,'Vehicle_Model_2Track_OLD'),1)))
    load_system('Vehicle_Model_2Track_OLD');
    open_system('Vehicle_Model_2Track_OLD');
end 