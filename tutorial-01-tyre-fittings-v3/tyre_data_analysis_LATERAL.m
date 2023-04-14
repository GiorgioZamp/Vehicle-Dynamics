%% Tyre Analysis for Lateral Forces
% Start importing datasets and preprocessing them to then compute lateral
% forces with Pure and Longitudinal Slip

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

% Conversion parameters 
to_rad = pi/180;
to_deg = 180/pi;

%% Select Tyre Dataset

% Dataset path
data_set_path = 'dataset\';
% dataset selection and loading
data_set = 'Hoosier_B1464run23'; % pure lateral forces (no k slip)
%{'Continental_B1464run8.mat','Continental_B1464run51.mat','Goodyear_B1464run13.mat','Goodyear_B1464run58.mat','Hoosier_B1464run30'}

fprintf('Loading dataset ...')

load ([data_set_path, data_set]); % pure lateral
cut_start = 27760;
cut_end   = 54500;
smpl_range = cut_start:cut_end;
diameter = 18; % inches

% Initialise tyre data
% THESE ARE GIVEN WITH TYRE SO BUILD A SWITCH ONCE YOU FIND THE VALUES
Fz0 = 220;   % [N] nominal load is given
R0  = diameter*2.56/2/100; % [m] get from nominal load R0 (m)
tyre_coeffs = initialise_tyre_data(R0, Fz0);

fprintf('completed!\n')

%% Plot Raw Data

figure
tiledlayout(6,1)

% Plot Normal Load
ax_list(1) = nexttile;
y_range = [min(min(-FZ),0) round(max(-FZ)*1.1)];
plot(-FZ)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')
ylim('padded')

% Plot Camber Angle
ax_list(2) = nexttile;
y_range = [min(min(IA),0) round(max(IA)*1.1)];
plot(IA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')
ylim('padded')

% Plot Side Slip Angle
ax_list(3) = nexttile; y_range = [min(min(SA),0) round(max(SA)*1.1)];
plot(SA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Side slip')
xlabel('Samples [-]')
ylabel('[deg]')
ylim('padded')

% Plot Longitudinal Slip
ax_list(4) = nexttile; y_range = [min(min(SL),0) round(max(SL)*1.1)];
plot(SL)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Longitudinal slip')
xlabel('Samples [-]')
ylabel('[-]')
ylim('padded')

% Plot Tyre Pressure
ax_list(5) = nexttile;
y_range = [min(min(P),0) round(max(P)*1.1)];
plot(P)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre pressure')
xlabel('Samples [-]')
ylabel('[psi]')
ylim('padded')

% Plot Tyre Temperature
ax_list(6) = nexttile;
y_range = [min(min(TSTC),0) round(max(TSTC)*1.1)];
plot(TSTC,'DisplayName','Center')
hold on
plot(TSTI,'DisplayName','Internal')
plot(TSTO,'DisplayName','Outboard')
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre temperatures')
xlabel('Samples [-]')
ylabel('[degC]')
ylim('padded')

linkaxes(ax_list,'x')

%% Select Data
% Cut data outside the specified sample range

% Create empty table
tyre_data = table();

% Store raw data in the table
tyre_data.SL =  SL(smpl_range);
tyre_data.SA =  SA(smpl_range)*to_rad;
tyre_data.FZ = -FZ(smpl_range);  % 0.453592  lb/kg
tyre_data.FX =  FX(smpl_range);
tyre_data.FY =  FY(smpl_range);
tyre_data.MZ =  MZ(smpl_range);
tyre_data.IA =  IA(smpl_range)*to_rad;

% Extract points at constant inclination angle
GAMMA_tol = 0.05*to_rad;
idx.GAMMA_0 = 0.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 0.0*to_rad+GAMMA_tol;
idx.GAMMA_1 = 1.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 1.0*to_rad+GAMMA_tol;
idx.GAMMA_2 = 2.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 2.0*to_rad+GAMMA_tol;
idx.GAMMA_3 = 3.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 3.0*to_rad+GAMMA_tol;
idx.GAMMA_4 = 4.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 4.0*to_rad+GAMMA_tol;
idx.GAMMA_5 = 5.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 5.0*to_rad+GAMMA_tol;
GAMMA_0  = tyre_data( idx.GAMMA_0, : );
GAMMA_1  = tyre_data( idx.GAMMA_1, : );
GAMMA_2  = tyre_data( idx.GAMMA_2, : );
GAMMA_3  = tyre_data( idx.GAMMA_3, : );
GAMMA_4  = tyre_data( idx.GAMMA_4, : );
GAMMA_5  = tyre_data( idx.GAMMA_5, : );

% Extract points at constant vertical load
FZ_tol = 100;
idx.FZ_220  = 220-FZ_tol < tyre_data.FZ & tyre_data.FZ < 220+FZ_tol;
idx.FZ_440  = 440-FZ_tol < tyre_data.FZ & tyre_data.FZ < 440+FZ_tol;
idx.FZ_700  = 700-FZ_tol < tyre_data.FZ & tyre_data.FZ < 700+FZ_tol;
idx.FZ_900  = 900-FZ_tol < tyre_data.FZ & tyre_data.FZ < 900+FZ_tol;
idx.FZ_1120 = 1120-FZ_tol < tyre_data.FZ & tyre_data.FZ < 1120+FZ_tol;
FZ_220  = tyre_data( idx.FZ_220, : );
FZ_440  = tyre_data( idx.FZ_440, : );
FZ_700  = tyre_data( idx.FZ_700, : );
FZ_900  = tyre_data( idx.FZ_900, : );
FZ_1120 = tyre_data( idx.FZ_1120, : );

% Extract Longitudinal Slip (HOW SINCE IT'S NULL)

% Plot Extracted Data
vec_samples = 1:1:length(smpl_range);

figure()
tiledlayout(2,1)

ax_list(1) = nexttile;
plot(tyre_data.IA*to_deg)
hold on
plot(vec_samples(idx.GAMMA_0),GAMMA_0.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_1),GAMMA_1.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_2),GAMMA_2.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_3),GAMMA_3.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_4),GAMMA_4.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_5),GAMMA_5.IA*to_deg,'.');
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')

ax_list(2) = nexttile;
plot(tyre_data.FZ)
hold on
plot(vec_samples(idx.FZ_220),FZ_220.FZ,'.');
plot(vec_samples(idx.FZ_440),FZ_440.FZ,'.');
plot(vec_samples(idx.FZ_700),FZ_700.FZ,'.');
plot(vec_samples(idx.FZ_900),FZ_900.FZ,'.');
plot(vec_samples(idx.FZ_1120),FZ_1120.FZ,'.');
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')

% plot(tyre_data.SA*to_deg)
% hold on
% plot(vec_samples(idx.SA_0),   SA_0.SA*to_deg,'.');
% plot(vec_samples(idx.SA_3neg),SA_3neg.SA*to_deg,'.');
% plot(vec_samples(idx.SA_6neg),SA_6neg.SA*to_deg,'.');
% title('Slide slip')
% xlabel('Samples [-]')
% ylabel('[rad]')

%% FITTING FOR PURE SLIP LATERAL FORCE
%  Nominal Load Fz==Fz0=220 N, Zero Camber, Longitudinal Slip k=0, p=80 psi

% Intersect tables to obtain specific sub-datasets
 % Extract data for zero slip and camber, and Nominal Load Fz0
[TData0, ~] = intersect_table_data( GAMMA_0, FZ_220 );

ALPHA_vec = TData0.SA; % extract for clarity
FY_vec    = TData0.FY;

% plot_selected_data
figure('Name','Selected-data')
plot_selected_data(TData0); % Fy has opposite sign w.r.t. side slip since the test was run in SAE ref frame

FZ0 = mean(TData0.FZ);

zeros_vec = zeros(size(TData0.SA));
ones_vec  = ones(size(TData0.SA));

% Guess
FY0_guess = MF96_FY0_vec(zeros_vec , ALPHA_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);

figure()
plot(ALPHA_vec,FY_vec,'o')
hold on
plot(ALPHA_vec,FY0_guess,'x')
hold off

% Guess values for parameters to be optimised
%    [𝗉𝖢𝗒𝟣, 𝗉𝖣𝗒𝟣, 𝗉𝖤𝗒𝟣, 𝗉𝖧𝗒𝟣, 𝗉𝖪𝗒𝟣, 𝗉𝖪𝗒𝟤, 𝗉𝖵𝗒1] 
P0 = [1,    1,   0,    0,    10,   0,    0];
lb = [  ];
ub = [  ];


SA_vec = -12*to_rad:0.001:12*to_rad; % side slip vector [rad]

% Optimize the coefficients
[P_fz_nom,~,~] = fmincon(@(P)resid_pure_Fy(P,FY_vec,ALPHA_vec,0,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values                             
tyre_coeffs.pCy1 = P_fz_nom(1) ;
tyre_coeffs.pDy1 = P_fz_nom(2) ;  
tyre_coeffs.pEy1 = P_fz_nom(3) ;
tyre_coeffs.pHy1 = P_fz_nom(4) ;
tyre_coeffs.pKy1 = P_fz_nom(5) ; 
tyre_coeffs.pKy2 = P_fz_nom(6) ;
tyre_coeffs.pVy1 = P_fz_nom(7) ;

% Use Magic Formula to compute the fitting function 
%FY0_fz_nom_vec = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, ...
                             % FZ0.*ones_vec,tyre_coeffs);
FY0_fz_nom_vec = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)), ...
                              FZ0.*ones(size(SA_vec)),tyre_coeffs);

% Plot Raw Data and Fitted Function
figure('Name','Fy0(Fz0)')
plot(ALPHA_vec,TData0.FY,'o')
hold on
plot(SA_vec,FY0_fz_nom_vec,'-','LineWidth',2)
xlabel('$\alpha$ [rad]')
ylabel('$F_{y0}$ [N]')
legend('Raw','Fitted')

%% Fit coefficient with VARIABLE LOAD Fz










