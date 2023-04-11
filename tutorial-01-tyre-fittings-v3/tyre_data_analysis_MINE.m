%% Tyre Analysis for Lateral Forces Fy
% Author: Zampieri Giorgio - Version: 08/04

%% Initialisation
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


to_rad = pi/180;
to_deg = 180/pi;

%% Select Tyre Dataset

% Dataset Path
data_set_path = 'TTC_dataset/';

% Dataset Selection and Loading
data_set = 'B1464run23'; % pure lateral forces + combined























%pCy1, pDy1, pEy1, pHy1, pKy1, pKy2, pVy1