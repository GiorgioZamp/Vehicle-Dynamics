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
data_set = 'Hoosier_B1464run23'; % pure lateral forces
%{'Continental_B1464run8.mat','Continental_B1464run51.mat','Goodyear_B1464run13.mat','Goodyear_B1464run58.mat','Hoosier_B1464run30'}

fprintf('Loading dataset ...')

load ([data_set_path, data_set]); % pure lateral
cut_start = 32380;
cut_end   = 54500;
smpl_range = cut_start:cut_end;
diameter = 18; % inches

% Initialise tyre data
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
tyre_data.FZ = -FZ(smpl_range); 
tyre_data.FX =  FX(smpl_range);
tyre_data.FY = -FY(smpl_range); % to ISO
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

% zeros_vec = zeros(size(TData0.SA));
% ones_vec  = ones(size(TData0.SA));
% Guess and plot it
% FY0_guess = MF96_FY0_vec(zeros_vec , ALPHA_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);
% figure()
% plot(ALPHA_vec,FY_vec,'o')
% hold on
% plot(ALPHA_vec,FY0_guess,'x')
% hold off

% Guess values for parameters to be optimised
%    [𝗉𝖢𝗒𝟣, 𝗉𝖣𝗒𝟣, 𝗉𝖤𝗒𝟣, 𝗉𝖧𝗒𝟣, 𝗉𝖪𝗒𝟣, 𝗉𝖪𝗒𝟤, 𝗉𝖵𝗒1] 
% P0 = [1,    1,   0,    0,    10,   0,    0]; % Good for SAE
P0 =   [1,1,1,1,1,1,1]; % Good for ISO
lb = [  ];
ub = [  ];


SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

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
FY0_fz_nom_vec = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)), ...
                              FZ0.*ones(size(SA_vec)),tyre_coeffs);

% Plot Raw Data and Fitted Function
figure('Name','Fy0(Fz0)')
plot(ALPHA_vec*to_deg,TData0.FY,'.')
hold on
plot(SA_vec*to_deg,FY0_fz_nom_vec,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')
legend('Raw','Fitted')

res_Fy0_nom  = resid_pure_Fy(P_fz_nom, FY_vec, ALPHA_vec, 0, FZ0, tyre_coeffs);
R2 = 1-res_Fy0_nom;
RMSE = sqrt(res_Fy0_nom*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
%err = [err ; R2 RMSE];
%% Fit coefficient with VARIABLE LOAD Fz

% Extract data with variable load
%[TDataDFz, ~] = intersect_table_data(SL_0, GAMMA_0);
TDataDFz = GAMMA_0; % since there's no long slip to intersect with

% Plot extracted data
% figure
% plot(TDataDFz.SA*to_deg,TDataDFz.FY,'.')

% Initialise values for parameters to be optimised
%    [pDy2,pEy2,pHy2,pVy2]
% P0 = [ -0.1072,   0.2531,  2.1548e-05,   7.2616e-04]; 
% lb = [ ];
% ub = [ ];
P0 = [-0.163509966342616,	-131.167452831061,	-0.000983362417156491,	-0.00912183395269232] ;
lb = [-1e3,-1e3,-1e3,-1e3];
ub = [ 1e2, 1e2, 1e2, 1e2];

ALPHA_vec = TDataDFz.SA;
FY_vec    = TDataDFz.FY;
FZ_vec    = TDataDFz.FZ;
SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

% Optimize the coefficients
[P_dfz,~,~] = fmincon(@(P)resid_pure_Fy_varFz(P,FY_vec,ALPHA_vec,0,FZ_vec,tyre_coeffs),...
               P0,[],[],[],[],lb,ub);

% GlobalSearch
% ms = MultiStart('FunctionTolerance',2e-4,'UseParallel',true);
% gs = GlobalSearch(ms);
% ResParms = @(P)resid_pure_Fy_varFz(P,FY_vec,ALPHA_vec,0,FZ_vec,tyre_coeffs);
% problem = createOptimProblem('fmincon','x0',[ -0.1072,   0.2531,  2.1548e-05,   7.2616e-04],...
%     'objective',ResParms,'lb',[],'ub',[]);
% P_dfz = run(gs,problem);

% MultiStart
% rng default % For reproducibility
% gs = GlobalSearch('FunctionTolerance',2e-4,'NumTrialPoints',2000);
% ms = MultiStart(gs,'UseParallel',true);
% ResParms = @(P)resid_pure_Fy_varFz(P,FY_vec,ALPHA_vec,0,FZ_vec,tyre_coeffs);
% opts = optimoptions(@fmincon,'Algorithm','sqp');
% problem = createOptimProblem('fmincon','objective',...
%     ResParms,'x0',[-0.107, 0.253, 2e-05, 8e-04],'lb',[],'ub',[],'options',opts);
% P_dfz = run(ms,problem,50);

% Update tyre data with new optimal values (change them also in resid_pure
% if you change the parameters
tyre_coeffs.pDy2 = P_dfz(1);
tyre_coeffs.pEy2 = P_dfz(2);
tyre_coeffs.pHy2 = P_dfz(3);
tyre_coeffs.pVy2 = P_dfz(4);



% Use Magic Formula to compute the fitting function 
FY0_fz_var_vec1 = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)),mean(FZ_220.FZ).*ones(size(SA_vec)),tyre_coeffs);
FY0_fz_var_vec2 = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)),mean(FZ_440.FZ).*ones(size(SA_vec)),tyre_coeffs);
FY0_fz_var_vec3 = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)),mean(FZ_700.FZ).*ones(size(SA_vec)),tyre_coeffs);
FY0_fz_var_vec4 = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)),mean(FZ_900.FZ).*ones(size(SA_vec)),tyre_coeffs);
FY0_fz_var_vec5 = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)),mean(FZ_1120.FZ).*ones(size(SA_vec)),tyre_coeffs);

% Plot Raw Data and Fitted Function
figure('Name','Fy0 vs Fz'), hold on, grid on;
plot(TDataDFz.SA*to_deg,TDataDFz.FY,'.')
plot(SA_vec*to_deg,FY0_fz_var_vec1,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec2,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec3,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec4,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec5,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$F_{y}$ [N]')

tmp = [220,440,700,900,1120];
leg = cell(length(tmp)+1,1);
leg{1} = 'Raw Data';
for i=1:length(tmp)
leg{i+1} = ['Fitted fz= ',num2str(tmp(i)),' N'];
end
legend(leg,Location="best")
hold off

res_Fy0_dfz  = resid_pure_Fy_varFz(P_dfz, FY_vec, ALPHA_vec, 0, FZ_vec, tyre_coeffs);
R2 = 1-res_Fy0_dfz;
RMSE = sqrt(res_Fy0_dfz*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
%err = [err ; R2 RMSE];

%% Cornering Stiffness
[alpha__y, By, Cy, Dy, Ey, ~, SVy] =MF96_FY0_coeffs(0, 0, 0, mean(FZ_220.FZ), tyre_coeffs);
Calfa_vec1_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
[alpha__y, By, Cy, Dy, Ey, ~, SVy] =MF96_FY0_coeffs(0, 0, 0, mean(FZ_440.FZ), tyre_coeffs);
Calfa_vec2_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
[alpha__y, By, Cy, Dy, Ey, ~, SVy] =MF96_FY0_coeffs(0, 0, 0, mean(FZ_700.FZ), tyre_coeffs);
Calfa_vec3_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
[alpha__y, By, Cy, Dy, Ey, ~, SVy] =MF96_FY0_coeffs(0, 0, 0, mean(FZ_900.FZ), tyre_coeffs);
Calfa_vec4_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
[alpha__y, By, Cy, Dy, Ey, ~, SVy] =MF96_FY0_coeffs(0, 0, 0, mean(FZ_1120.FZ), tyre_coeffs);
Calfa_vec5_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);

tmp_zeros = zeros(size(SA_vec));
tmp_ones = ones(size(SA_vec));

Calfa_vec1 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec2 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec3 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec4 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec5 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

figure('Name','C_alpha')
subplot(2,1,1)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(mean(FZ_220.FZ),Calfa_vec1_0,'+','LineWidth',2)
plot(mean(FZ_440.FZ),Calfa_vec2_0,'+','LineWidth',2)
plot(mean(FZ_700.FZ),Calfa_vec3_0,'+','LineWidth',2)
plot(mean(FZ_900.FZ),Calfa_vec4_0,'+','LineWidth',2)
plot(mean(FZ_1120.FZ),Calfa_vec5_0,'+','LineWidth',2)
xlabel('$F_z (N)$')
ylabel('$C_{\alpha} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

subplot(2,1,2)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(SA_vec,Calfa_vec1,'-','LineWidth',2)
plot(SA_vec,Calfa_vec2,'-','LineWidth',2)
plot(SA_vec,Calfa_vec3,'-','LineWidth',2)
plot(SA_vec,Calfa_vec4,'-','LineWidth',2)
plot(SA_vec,Calfa_vec5,'-','LineWidth',2)
xlabel('$\alpha (rad)$')
ylabel('$C_{\alpha} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

%% Fit coefficient with VARIABLE CAMBER
% Zero longitudinal slip k and fixed normal load Fz

% Extract data with variable camber
%[TDataGamma, ~] = intersect_table_data(SL_0,FZ_220);
TDataGamma = FZ_220; % since SL is already zero everywhere

% Initialise values for parameters to be optimised
%    [pDy3,pEy3,pEy4,pHy3,pKy3,pVy3,pVy4]
P0 = [ 1,   1,   1,   1,   1,   1,   1  ]; 
lb = [];
ub = [];

ALPHA_vec = TDataGamma.SA; % extract for clarity
GAMMA_vec = TDataGamma.IA;
FY_vec    = TDataGamma.FY;
FZ0 = mean(TDataGamma.FZ);
SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

% Optimize the coefficients (FIT WITH alpha=0?)
[P_varGamma,~,~] = fmincon(@(P)resid_pure_Fy_varGamma(P,FY_vec,ALPHA_vec,GAMMA_vec,FZ0,tyre_coeffs),...
               P0,[],[],[],[],lb,ub);

% Change tyre data with new optimal values 
tyre_coeffs.pDy3 = P_varGamma(1);
tyre_coeffs.pEy3 = P_varGamma(2);
tyre_coeffs.pEy4 = P_varGamma(3);
tyre_coeffs.pHy3 = P_varGamma(4);
tyre_coeffs.pKy3 = P_varGamma(5);
tyre_coeffs.pVy3 = P_varGamma(6);
tyre_coeffs.pVy4 = P_varGamma(7);

% Use Magic Formula to compute the fitting function 
zeros_vec = zeros(size(SA_vec));
ones_vec  = ones(size(SA_vec));

FY0_varGamma_vec1 = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_0.IA).*ones_vec, FZ0*ones_vec,tyre_coeffs);
FY0_varGamma_vec2 = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_1.IA).*ones_vec, FZ0*ones_vec,tyre_coeffs);
FY0_varGamma_vec3 = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_2.IA).*ones_vec, FZ0*ones_vec,tyre_coeffs);
FY0_varGamma_vec4 = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_3.IA).*ones_vec, FZ0*ones_vec,tyre_coeffs);
FY0_varGamma_vec5 = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_4.IA).*ones_vec, FZ0*ones_vec,tyre_coeffs);

% Plot Raw Data and Fitted Function
figure('Name','Fy0 vs Gamma')
plot(ALPHA_vec*to_deg,TDataGamma.FY,'.')
hold on
plot(SA_vec*to_deg,FY0_varGamma_vec1,'-')
plot(SA_vec*to_deg,FY0_varGamma_vec2,'-')
plot(SA_vec*to_deg,FY0_varGamma_vec3,'-')
plot(SA_vec*to_deg,FY0_varGamma_vec4,'-')
plot(SA_vec*to_deg,FY0_varGamma_vec5,'-')
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')
tmp = [0,1,2,3,4];
leg = cell(length(tmp)+1,1);
leg{1} = 'Raw Data';
for i=1:length(tmp)
leg{i+1} = ['Fitted $\gamma$ = ',num2str(tmp(i)),' [°]'];
end
legend(leg,Location="best")
hold off

res_Fy0_gamm  = resid_pure_Fy_varGamma(P_varGamma, FY_vec, ALPHA_vec,GAMMA_vec, FZ0, tyre_coeffs);
R2 = 1-res_Fy0_gamm;
RMSE = sqrt(res_Fy0_gamm*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
%err = [err ; R2 RMSE];

%% Fit Self Aligning Moment

[TDataMz, ~] = intersect_table_data( GAMMA_0, FZ_220 );

ALPHA_vec = TDataMz.SA;
MZ_vec    = TDataMz.MZ;

FZ0 = mean(TDataMz.FZ);

% Guess values for parameters to be optimised
%    [qBz1,qBz9,qBz10,qCz1,qDz1,qDz6,qEz1,qEz4,qHz1]
P0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1  ];
lb = [  ];
ub = [  ];

% Optimize the coefficients
[P_Mz,~,~] = fmincon(@(P)resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% MultiStart (to find right P0-> then switch to fmincon)
% rng default % For reproducibility
% gs = GlobalSearch('FunctionTolerance',2e-4,'NumTrialPoints',2000);
% ms = MultiStart(gs,'UseParallel',true);
% ResParms = @(P)resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0,tyre_coeffs);
% opts = optimoptions(@fmincon,'Algorithm','sqp');
% problem = createOptimProblem('fmincon','objective',...
%     ResParms,'x0',P0,'lb',lb,'ub',ub,'options',opts);
% P_Mz = run(ms,problem,50);

% Update tyre data with new optimal values                             
tyre_coeffs.qBz1  = P_Mz(1) ;
tyre_coeffs.qBz9  = P_Mz(2) ;  
tyre_coeffs.qBz10 = P_Mz(3) ;
tyre_coeffs.qCz1  = P_Mz(4) ;
tyre_coeffs.qDz1  = P_Mz(5) ; 
tyre_coeffs.qDz6  = P_Mz(6) ;
tyre_coeffs.qEz1  = P_Mz(7) ;
tyre_coeffs.qEz4  = P_Mz(8) ;
tyre_coeffs.qHz1  = P_Mz(9) ;

SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

Fy = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)), FZ0.*ones(size(SA_vec)),tyre_coeffs);

Mz0_fit = MF96_Mz0_vec(zeros(size(SA_vec)), SA_vec,zeros(size(SA_vec)), FZ0*ones(size(SA_vec)), Fy, tyre_coeffs);


% Plot Raw Data and Fitted Function
figure('Name','Mz0(Fz0)')
plot(ALPHA_vec*180/pi,TDataMz.MZ,'.')
hold on
plot(SA_vec*180/pi,Mz0_fit,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend('Raw','Fitted')

res_Mz0  = resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0,tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% Self Aligning Moment with Variable Fz

% [TDataMz_dFz, ~] = intersect_table_data(KAPPA_0 GAMMA_0);
TDataMz_dFz = GAMMA_0;
ALPHA_vec   = TDataMz_dFz.SA;
FY_vec      = TDataMz_dFz.FY;
FZ_vec      = TDataMz_dFz.FZ; 
MZ_vec      = TDataMz_dFz.MZ;
zeros_vec = zeros(size(ALPHA_vec));
ones_vec = ones(size(ALPHA_vec));

% Guess values for parameters to be optimised
%    [qHz2, qBz2, qBz3, qDz2, qEz2, qEz3, qDz7]
P0 = [-1, 1, -1, 1, -1, 1, -1] ;
lb = [];
ub = [];

% Optimize the coefficients 
[P_Mz_varFz,~,~] = fmincon(@(P)resid_pure_Mz_varFz(P, MZ_vec, ALPHA_vec, zeros_vec, FZ_vec, tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% MultiStart (to find right P0-> then switch to fmincon)
% rng default % For reproducibility
% gs = GlobalSearch('FunctionTolerance',2e-4,'NumTrialPoints',2000);
% ms = MultiStart(gs,'UseParallel',true);
% ResParms = @(P)resid_pure_Mz_varFz(P, MZ_vec, ALPHA_vec, zeros_vec, FZ_vec, tyre_coeffs);
% opts = optimoptions(@fmincon,'Algorithm','sqp');
% problem = createOptimProblem('fmincon','objective',...
%     ResParms,'x0',P0,'lb',lb,'ub',ub,'options',opts);
% P_Mz_varFz = run(ms,problem,50);

% Update tyre data with new optimal values                             
tyre_coeffs.qHz2  = P_Mz_varFz(1) ;
tyre_coeffs.qBz2  = P_Mz_varFz(2) ;  
tyre_coeffs.qBz3  = P_Mz_varFz(3) ;
tyre_coeffs.qDz2  = P_Mz_varFz(4) ;
tyre_coeffs.qEz2  = P_Mz_varFz(5) ; 
tyre_coeffs.qEz3  = P_Mz_varFz(6) ;
tyre_coeffs.qDz7  = P_Mz_varFz(7) ;

%SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

Fy = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_220.FZ)*ones_vec,tyre_coeffs);
Mz0_220 = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_220.FZ)*ones_vec, Fy, tyre_coeffs);
Fy = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_440.FZ)*ones_vec,tyre_coeffs);
Mz0_440 = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_440.FZ)*ones_vec, Fy, tyre_coeffs);
Fy = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_700.FZ)*ones_vec,tyre_coeffs);
Mz0_700 = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_700.FZ)*ones_vec, Fy, tyre_coeffs);
Fy = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_900.FZ)*ones_vec,tyre_coeffs);
Mz0_900 = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_900.FZ)*ones_vec, Fy, tyre_coeffs);
Fy = MF96_FY0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_1120.FZ)*ones_vec,tyre_coeffs);
Mz0_1120 = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, mean(FZ_1120.FZ)*ones_vec, Fy, tyre_coeffs);


% Plot Raw Data and Fitted Function
figure('Name','Mz0(Fz)'), hold on;
plot(ALPHA_vec*to_deg,MZ_vec,'.')
plot(ALPHA_vec*to_deg,Mz0_220,'-','LineWidth',2)
plot(ALPHA_vec*to_deg,Mz0_440,'-','LineWidth',2)
plot(ALPHA_vec*to_deg,Mz0_700,'-','LineWidth',2)
plot(ALPHA_vec*to_deg,Mz0_900,'-','LineWidth',2)
plot(ALPHA_vec*to_deg,Mz0_1120,'-','LineWidth',2)

leg = cell(1,6);
leg{1} = 'Raw Data';
tmp = [220,440,700,900,1120];
for i = 1:5
    leg{i+1} = ['Fitted fz= ',num2str(tmp(i)),' N'];
end
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend(leg,Location='Best')

res_Mz0  = resid_pure_Mz_varFz(P, MZ_vec, ALPHA_vec, zeros_vec, FZ_vec, tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
% err = [err ; R2 RMSE];

%% Self Aligning Moment with Variable Camber

% [TDataMz_tmp, ~] = intersect_table_data(KAPPA_0, FZ_220 );
TDataMz_tmp = FZ_220;
ALPHA_vec   = TDataMz_tmp.SA;
FY_vec      = TDataMz_tmp.FY;
GAMMA_vec   = TDataMz_tmp.IA;
MZ_vec      = TDataMz_tmp.MZ;
FZ_vec      = TDataMz_tmp.FZ;
zeros_vec = zeros(size(ALPHA_vec));
ones_vec = ones(size(ALPHA_vec));

% Guess values for parameters to be optimised
%  [qHz3, qHz4, qBz4, qBz5, qDz3, qDz4,qEz5, qDz8, qDz9]
P0 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
lb = [];
ub = [];

% Optimize the coefficients 
[P_Mz_gamma,~,~] = fmincon(@(P)resid_pure_Mz_gamma(P, MZ_vec, ALPHA_vec, GAMMA_vec, FZ_vec, tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values                             
tyre_coeffs.qHz3  = P_Mz_gamma(1) ;
tyre_coeffs.qHz4  = P_Mz_gamma(2) ;  
tyre_coeffs.qBz4  = P_Mz_gamma(3) ;
tyre_coeffs.qBz5  = P_Mz_gamma(4) ;
tyre_coeffs.qDz3  = P_Mz_gamma(5) ; 
tyre_coeffs.qDz4  = P_Mz_gamma(6) ;
tyre_coeffs.qEz5  = P_Mz_gamma(7) ;
tyre_coeffs.qDz8  = P_Mz_gamma(8) ;
tyre_coeffs.qDz9  = P_Mz_gamma(9) ;

SA_vec = min(ALPHA_vec):1e-4:max(ALPHA_vec);
zeros_vec = zeros(size(SA_vec));
ones_vec = ones(size(SA_vec));
FZ_vec = mean(TDataMz_tmp.FZ)*ones_vec;

Fy = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_0.IA)*ones_vec, FZ_vec,tyre_coeffs);
Mz0_0 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_0.IA)*ones_vec, FZ_vec, Fy, tyre_coeffs);

Fy = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_1.IA)*ones_vec, FZ_vec,tyre_coeffs);
Mz0_1 = MF96_Mz0_vec(zeros_vec, SA_vec,mean(GAMMA_1.IA)*ones_vec, FZ_vec, Fy, tyre_coeffs);

Fy = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_2.IA)*ones_vec, FZ_vec,tyre_coeffs);
Mz0_2 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_2.IA)*ones_vec, FZ_vec, Fy, tyre_coeffs);

Fy = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_3.IA)*ones_vec, FZ_vec,tyre_coeffs);
Mz0_3 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_3.IA)*ones_vec, FZ_vec, Fy, tyre_coeffs);

Fy = MF96_FY0_vec(zeros_vec, SA_vec, mean(GAMMA_4.IA)*ones_vec, FZ_vec,tyre_coeffs);
Mz0_4 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_4.IA)*ones_vec, FZ_vec, Fy, tyre_coeffs);


% Plot Raw Data and Fitted Function
figure('Name','Mz0($\gamma$)'), hold on;
plot(ALPHA_vec*to_deg,MZ_vec,'.')
plot(SA_vec*to_deg,Mz0_0,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_1,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_2,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_3,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_4,'-','LineWidth',2)

leg = cell(1,6);
leg{1} = 'Raw Data';
tmp = [0,1,2,4,5];
for i = 1:5
    leg{i+1} = ['Fitted $\gamma$= ',num2str(tmp(i)),' °'];
end
xlabel('$\gamma$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend(leg,Location='Best')

res_Mz0  = resid_pure_Mz_gamma(P, MZ_vec, ALPHA_vec, GAMMA_vec, FZ_vec, tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
% err = [err ; R2 RMSE];

%% Extract Coefficients
% Save tyre data structure to mat file
save(['tyre_' data_set,'.mat'],'tyre_coeffs');









