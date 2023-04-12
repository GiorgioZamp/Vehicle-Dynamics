%% Tyre Analysis for Lateral Forces
% Start importing datasets and preprocessing them to then compute lateral
% forces with Pure and Longitudinal Slip

%% INITIALISATION
clc
clearvars 
close all   

% Set LaTeX as default interpreter for axis labels, ticks and legends
set(0,'defaulttextinterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

set(0,'DefaultFigureWindowStyle','docked');
set(0,'defaultAxesFontSize',  16)
set(0,'DefaultLegendFontSize',16)

addpath('tyre_lib/')

% Conversion parameters 
to_rad = pi/180;
to_deg = 180/pi;

%% Select Tyre Dataset

% Dataset path
data_set_path = 'dataset/';
% dataset selection and loading
data_set = 'Hoosier_B1464run23'; % pure lateral forces + combined

fprintf('Loading dataset ...')

load ([data_set_path, 'B1464run23.mat']); % pure lateral
cut_start = 27760;
cut_end   = 54500;
smpl_range = cut_start:cut_end;

fprintf('completed!\n')

%% Plot Raw Data