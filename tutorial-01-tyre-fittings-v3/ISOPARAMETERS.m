%% Initialisation
clc
clearvars
close all

% Set LaTeX as default interpreter for axis labels, ticks and legends
set(0,'defaulttextinterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesXGrid','on')
set(groot,'defaultAxesYGrid','on')

set(0,'DefaultFigureWindowStyle','docked');
set(0,'defaultAxesFontSize',  16)
set(0,'DefaultLegendFontSize',16)

addpath('tyre_lib/','dataset\')

to_rad = pi/180;
to_deg = 180/pi;

%% Select tyre dataset: -> LATERAL

% dataset path
data_set_path = 'dataset/';

% dataset selection and loading
data_set = 'Hoosier_B1464run23'; % pure lateral forces
% data_set = 'Hoosier_B1464run30';  % braking/traction (pure long. force) + combined

% tyre geometric data:
% Hoosier	18.0x6.0-10
% 18 diameter in inches
% 6.0 section width in inches
% tread width in inches
diameter = 18*2.56;   % Converting inches to cm
Fz0 = 220;            % [N] nominal load is given
R0  = diameter/2/100; % [m] get from nominal load R0 (m)


fprintf('Loading dataset ...')
switch data_set
    case 'Hoosier_B1464run23'
       load ([data_set_path,data_set, '.mat']); % pure lateral
       cut_start = 32380;
       cut_end   = 54500;
    case 'Hoosier_B1464run30'
        load ([data_set_path,data_set,'.mat']); % pure longitudinal
        cut_start = 19028;
        cut_end   = 37643;
    otherwise
        error('Not found dataset: `%s`\n', data_set) ;
end


% select dataset portion
smpl_range = cut_start:cut_end;

fprintf('completed!\n')

%% Plot Raw Data

figure
tiledlayout(6,1)

ax_list(1) = nexttile; y_range = [min(min(-FZ),0) round(max(-FZ)*1.1)];
plot(-FZ)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')

ax_list(2) = nexttile; y_range = [min(min(IA),0) round(max(IA)*1.1)];
plot(IA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')
ylim('padded')

ax_list(3) = nexttile; y_range = [min(min(SA),0) round(max(SA)*1.1)];
plot(SA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Side slip')
xlabel('Samples [-]')
ylabel('[deg]')

ax_list(4) = nexttile; y_range = [min(min(SL),0) round(max(SL)*1.1)];
plot(SL)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Longitudinal slip')
xlabel('Samples [-]')
ylabel('[-]')

ax_list(5) = nexttile; y_range = [min(min(P),0) round(max(P)*1.1)];
plot(P)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre pressure')
xlabel('Samples [-]')
ylabel('[psi]')

ax_list(6) = nexttile;  y_range = [min(min(TSTC),0) round(max(TSTC)*1.1)];
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

linkaxes(ax_list,'x')

%% Select some specific data
% Cut crappy data and select only 12 psi data

vec_samples = 1:1:length(smpl_range);
tyre_data = table(); % create empty table

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store raw data in the table
tyre_data.SL =  SL(smpl_range);                    %Slip Ratio based on RE (Longi.)
tyre_data.SA =  SA(smpl_range)*to_rad;             %Slip angle (Lateral)
tyre_data.FZ = -FZ(smpl_range);  % 0.453592  lb/kg %Vertical Load
tyre_data.FX =  FX(smpl_range);                    %Longitudinal Force
tyre_data.FY =  FY(smpl_range);                    %Lateral Force
tyre_data.MZ =  MZ(smpl_range);                    %Self Aliging Moments
tyre_data.IA =  IA(smpl_range)*to_rad;             %Inclination Angle (Camber)
tyre_data.P  =  P(smpl_range);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
% Test data done at:
%  - 50lbf  ( 50*0.453592*9.81 =  223N )
%  - 150lbf (150*0.453592*9.81 =  667N )
%  - 200lbf (200*0.453592*9.81 =  890N )
%  - 250lbf (250*0.453592*9.81 = 1120N )

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

% The slip angle is varied continuously between -4 and +12° and then
% between -12° and +4° for the pure slip case

P_tol = 10;
idx.P_80 = 80-P_tol<tyre_data.P & tyre_data.P<80+P_tol;
P_80 = tyre_data(idx.P_80,:);

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


%% FY0 - PURE SLIP LATERAL FORCE
%  Nominal Load Fz==Fz0=220 N, Zero Camber, Longitudinal Slip k=0, p=80 psi

% initialise tyre data
tyre_coeffs = initialise_tyre_dataTMP(R0, Fz0);

% Intersect tables to obtain specific sub-datasets
 % Extract data for zero slip and camber, and Nominal Load Fz0
[TDataTmp, ~] = intersect_table_data( GAMMA_0, FZ_220 );

ALPHA_vec = TDataTmp.SA; % extract for clarity
FY_vec    = TDataTmp.FY;

% plot_selected_data
figure('Name','Selected-data')
plot_selected_data(TDataTmp); % Fy has opposite sign w.r.t. side slip since the test was run in SAE ref frame

FZ0 = mean(TDataTmp.FZ);

% Guess values for parameters to be optimised
%    [𝗉𝖢𝗒𝟣, 𝗉𝖣𝗒𝟣,  𝗉𝖤𝗒𝟣,    𝗉𝖧𝗒𝟣,      𝗉𝖪𝗒𝟣,     𝗉𝖪𝗒𝟤,    𝗉𝖵𝗒1] 
% P0 = [1.12541180337932	2.71903809386550	0.443814145673160	-0.00380219704257947	-110.004170170015	3.08313490429168	0.0792154945741895];
P0 = [1.55  2.6 0.42 0    -132 5 0.5];
% P0 = [1.3  2.6168  0      0    -132.2724 4.9992 0.5];
% lb = [1,    -inf, -inf, -inf, -inf, 1.5, -inf]; 
% ub = [+inf, +inf, 1,    +inf, +inf, 3,   +inf];
lb = [];
ub = [];

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
f = figure('Name','Fy0(Fz0)');
plot(ALPHA_vec*to_deg,TDataTmp.FY,'.')
hold on
plot(SA_vec*to_deg,FY0_fz_nom_vec,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')
legend('Raw','Fitted')
title('Pure Slip Lateral Force')
exportgraphics(f,'GraphsISO/Fy0.eps')

res_Fy0_nom  = resid_pure_Fy(P_fz_nom, FY_vec, ALPHA_vec, 0, FZ0, tyre_coeffs);
R2 = 1-res_Fy0_nom;
RMSE = sqrt(res_Fy0_nom*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [];
err = [err ; R2 RMSE];

%% FY0_dfz - LATERAL FORCE WITH VARIABLE LOAD FZ

% Extract data with variable load
TDataTmp = intersect_table_data(GAMMA_0, P_80);

% Plot extracted data
% figure
% plot(TDataDFz.SA*to_deg,TDataDFz.FY,'.')

% Initialise values for parameters to be optimised
%    [pDy2,pEy2,pHy2,pVy2]
P0 = [0 0 0 0];
lb = [-0.008 0.07 -1 0.007];%lb = [-0.008 0.07 -1 0.007];rmse:66.661
ub = [ 1 1 1 1];

ALPHA_vec = TDataTmp.SA;
FY_vec    = TDataTmp.FY;
FZ_vec    = TDataTmp.FZ;
SA_vec = min(ALPHA_vec):1e-4:max(ALPHA_vec); % side slip vector [rad]

% Optimize the coefficients
P_dfz = fmincon(@(P)resid_pure_Fy_varFz(P,FY_vec,ALPHA_vec,0,FZ_vec,tyre_coeffs),...
               P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values
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
f = figure('Name','Fy0 vs Fz');
hold on
plot(TDataTmp.SA*to_deg,TDataTmp.FY,'.')
plot(SA_vec*to_deg,FY0_fz_var_vec1,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec2,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec3,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec4,'-','LineWidth',2)
plot(SA_vec*to_deg,FY0_fz_var_vec5,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')

tmp = [220,440,700,900,1120];
leg = cell(length(tmp)+1,1);
leg{1} = 'Raw Data';
for i=1:length(tmp)
leg{i+1} = ['Fitted fz= ',num2str(tmp(i)),' N'];
end
legend(leg,Location="best")
hold off
exportgraphics(f,'GraphsISO/Fy0_dFz.eps')

res_Fy0_dfz  = resid_pure_Fy_varFz(P_dfz, FY_vec, ALPHA_vec, 0, FZ_vec, tyre_coeffs);
R2 = 1-res_Fy0_dfz;
RMSE = sqrt(res_Fy0_dfz*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% Cy_alpha - Cornering Stiffness
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
Calfa_vec2 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_440.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec3 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec4 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec5 = MF96_CorneringStiffnessFY(tmp_zeros, SA_vec ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

f = figure('Name','C_alpha');
subplot(2,1,1)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(mean(FZ_220.FZ),-Calfa_vec1_0,'+','LineWidth',2)
plot(mean(FZ_440.FZ),-Calfa_vec2_0,'+','LineWidth',2)
plot(mean(FZ_700.FZ),-Calfa_vec3_0,'+','LineWidth',2)
plot(mean(FZ_900.FZ),-Calfa_vec4_0,'+','LineWidth',2)
plot(mean(FZ_1120.FZ),-Calfa_vec5_0,'+','LineWidth',2)
xlabel('$F_z (N)$')
ylabel('$C_{\alpha} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

subplot(2,1,2)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(SA_vec,-Calfa_vec1,'-','LineWidth',2)
plot(SA_vec,-Calfa_vec2,'-','LineWidth',2)
plot(SA_vec,-Calfa_vec3,'-','LineWidth',2)
plot(SA_vec,-Calfa_vec4,'-','LineWidth',2)
plot(SA_vec,-Calfa_vec5,'-','LineWidth',2)
xlabel('$\alpha (rad)$')
ylabel('$C_{\alpha} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

exportgraphics(f,'GraphsISO/C_alpha.eps')
%% FY0_gamma - LATERAL FORCE with VARIABLE CAMBER
% Zero longitudinal slip k and fixed normal load Fz

% Extract data with variable camber
TDataTmp = intersect_table_data(FZ_900,P_80); 

% Initialise values for parameters to be optimised
%    [pDy3,pEy3,pEy4,pHy3,pKy3,pVy3,pVy4]
P0 = [0,0,0,0,0,0,0]; 
lb = [];
ub = [];

% Cut messy parts
% cut = [104:840,990:1725,1873:2609,2761:3495,3645:4380];

ALPHA_vec = TDataTmp.SA; % extract for clarity
GAMMA_vec = TDataTmp.IA;
FY_vec    = TDataTmp.FY;
FZ0 = mean(TDataTmp.FZ);
SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

% Optimize the coefficients 
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
f = figure('Name','Fy0 vs Gamma');
plot(ALPHA_vec*to_deg,FY_vec,'.')
hold on
plot(SA_vec*to_deg,FY0_varGamma_vec1,'-','LineWidth',1.5)
plot(SA_vec*to_deg,FY0_varGamma_vec2,'-','LineWidth',1.5)
plot(SA_vec*to_deg,FY0_varGamma_vec3,'-','LineWidth',1.5)
plot(SA_vec*to_deg,FY0_varGamma_vec4,'-','LineWidth',1.5)
plot(SA_vec*to_deg,FY0_varGamma_vec5,'-','LineWidth',1.5)
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')
tmp = [0,1,2,3,4];
leg = cell(length(tmp)+1,1);
leg{1} = 'Raw Data';
for i=1:length(tmp)
leg{i+1} = ['Fitted ','$\gamma$',' = ',num2str(tmp(i)),' [deg]'];
end
legend(leg,Location="best")
hold off
exportgraphics(f,'GraphsISO/Fy0_gamma.eps')

res_Fy0_gamm  = resid_pure_Fy_varGamma(P_varGamma, FY_vec, ALPHA_vec, GAMMA_vec, FZ0, tyre_coeffs);
R2 = 1-res_Fy0_gamm;
RMSE = sqrt(res_Fy0_gamm*sum(FY_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% MZ - PURE SLIP SELF ALIGNING MOMENT

[TDataTmp, ~] = intersect_table_data( GAMMA_0, FZ_220 );

cut = 104:840;
ALPHA_vec = TDataTmp.SA(cut);
MZ_vec    = TDataTmp.MZ(cut);
FZ0       = mean(TDataTmp.FZ(cut));

% Guess values for parameters to be optimised
%    [qBz1,qBz9,qBz10,qCz1,qDz1,qDz6,qEz1,qEz4,qHz1]
P0 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1];
lb = [0,-1,-15,14,  0,14,-200,0,-1];
ub = [1, 0,-10,20,  1,18,0,1,0];

% Optimize the coefficients
[P_Mz,~,~] = fmincon(@(P)resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

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

%SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]
zeros_vec = zeros(size(ALPHA_vec));
ones_vec = ones(size(ALPHA_vec));

Mz0_fit = MF96_Mz0_vec(zeros_vec, ALPHA_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);

% Plot Raw Data and Fitted Function
f = figure('Name','Mz0(Fz0)');
plot(ALPHA_vec*to_deg,MZ_vec,'.')
hold on
plot(ALPHA_vec*to_deg,Mz0_fit,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend('Raw','Fitted')
exportgraphics(f,'GraphsISO/Mz0.eps')

res_Mz0  = resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0,tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% MZ_dFz - Self Aligning Moment with Variable Fz      !!!NEED FIXING!!!

TDataTmp    = intersect_table_data(GAMMA_0,P_80);
cut = [70:837, 978:1722, 1820:2607, 2757:3492, 3586:4377];
ALPHA_vec   = TDataTmp.SA(cut);
FZ_vec      = TDataTmp.FZ(cut); 
MZ_vec      = TDataTmp.MZ(cut);
zeros_vec = zeros(size(ALPHA_vec));
% cut bad parts
idx = ALPHA_vec>-12*to_rad & ALPHA_vec<12*to_rad;
MZ_vec    = MZ_vec(idx,:);
ALPHA_vec = ALPHA_vec(idx,:);
FZ_vec    = FZ_vec(idx,:);

% Guess values for parameters to be optimised
%    [qHz2, qBz2, qBz3, qDz2, qEz2, qEz3, qDz7]
P0 = [0 0 0 0 0 0 0];
lb = [-1 -1 -1 -1 -1 -1 -4];
ub = [1 1 1 1 1 1 1];

% Optimize the coefficients 
[P_Mz_varFz,~,~] = fmincon(@(P)resid_pure_Mz_varFz(P, MZ_vec, ALPHA_vec, zeros_vec, FZ_vec, tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values                             
tyre_coeffs.qHz2  = P_Mz_varFz(1) ;
tyre_coeffs.qBz2  = P_Mz_varFz(2) ;  
tyre_coeffs.qBz3  = P_Mz_varFz(3) ;
tyre_coeffs.qDz2  = P_Mz_varFz(4) ;
tyre_coeffs.qEz2  = P_Mz_varFz(5) ; 
tyre_coeffs.qEz3  = P_Mz_varFz(6) ;
tyre_coeffs.qDz7  = P_Mz_varFz(7) ;

SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]
zeros_vec = zeros(size(SA_vec));
ones_vec = ones(size(SA_vec));

Mz0_220 = MF96_Mz0_vec(zeros_vec, SA_vec, zeros_vec, mean(FZ_220.FZ)*ones_vec, tyre_coeffs);
Mz0_440 = MF96_Mz0_vec(zeros_vec, SA_vec, zeros_vec, mean(FZ_440.FZ)*ones_vec, tyre_coeffs);
Mz0_700 = MF96_Mz0_vec(zeros_vec, SA_vec, zeros_vec, mean(FZ_700.FZ)*ones_vec, tyre_coeffs);
Mz0_900 = MF96_Mz0_vec(zeros_vec, SA_vec, zeros_vec, mean(FZ_900.FZ)*ones_vec, tyre_coeffs);
Mz0_1120 = MF96_Mz0_vec(zeros_vec, SA_vec, zeros_vec, mean(FZ_1120.FZ)*ones_vec, tyre_coeffs);


% Plot Raw Data and Fitted Function
f = figure('Name','Mz0(Fz)');
hold on;
plot(ALPHA_vec*to_deg,MZ_vec,'.')
plot(SA_vec*to_deg,Mz0_220,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_440,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_700,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_900,'-','LineWidth',2)
plot(SA_vec*to_deg,Mz0_1120,'-','LineWidth',2)

leg = cell(1,6);
leg{1} = 'Raw Data';
tmp = [220,440,700,900,1120];
for i = 1:5
    leg{i+1} = ['Fitted fz= ',num2str(tmp(i)),' N'];
end
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend(leg,Location='Best')
exportgraphics(f,'GraphsISO/Mz0_dFz.eps')

res_Mz0  = resid_pure_Mz_varFz(P, MZ_vec, ALPHA_vec, zeros(size(ALPHA_vec)), FZ_vec, tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% MZ_gamma - Self Aligning Moment with Variable Camber 

% [TDataMz_tmp, ~] = intersect_table_data(KAPPA_0, FZ_220 );
TDataTmp = FZ_220;
cut = [104:840,990:1725,1873:2609,2761:3495,3645:4380];
ALPHA_vec   = TDataTmp.SA(cut);
GAMMA_vec   = TDataTmp.IA(cut);
MZ_vec      = TDataTmp.MZ(cut);
FZ_vec      = TDataTmp.FZ(cut);

% Guess values for parameters to be optimised
%    [qHz3, qHz4, qBz4, qBz5, qDz3, qDz4,qEz5, qDz8, qDz9]
P0 = [0.72,0.09,-0.28,-0.28,-3.32,42.54,-9.69,-1.83,-0.80];
lb = [-1,-1,-10,-inf,-inf,-inf,-inf,-inf,-inf];
ub = [+1, 1, 10, inf, inf, inf, inf, inf, inf];

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
FZ_vec = mean(FZ_vec)*ones_vec;

Mz0_0 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_0.IA)*ones_vec, FZ_vec, tyre_coeffs);
Mz0_1 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_1.IA)*ones_vec, FZ_vec, tyre_coeffs);
Mz0_2 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_2.IA)*ones_vec, FZ_vec, tyre_coeffs);
Mz0_3 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_3.IA)*ones_vec, FZ_vec, tyre_coeffs);
Mz0_4 = MF96_Mz0_vec(zeros_vec, SA_vec, mean(GAMMA_4.IA)*ones_vec, FZ_vec, tyre_coeffs);

% Plot Raw Data and Fitted Function
f = figure('Name','Mz0(\gamma)');
hold on
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
    leg{i+1} = ['Fitted ','$\gamma$',' = ',num2str(tmp(i)),' deg'];
end
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend(leg,Location='Best')
exportgraphics(f,'GraphsISO/Mz0_gamma.eps')

res_Mz0  = resid_pure_Mz_gamma(P, MZ_vec, ALPHA_vec, GAMMA_vec, FZ_vec, tyre_coeffs);
R2 = 1-res_Mz0;
RMSE = sqrt(res_Mz0*sum(MZ_vec.^2)/length(ALPHA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% LOAD COMBINED DATASET-------------------------------------------------------------
disp('Switchig to Braking/Traction dataset')
clearvars -except tyre_coeffs err
%---------------------------------------------------------------------------------------
%% Select tyre dataset: -> BRAKING/TRACTION + COMBINED SLIP

% dataset path
data_set_path = 'dataset/';

% dataset selection and loading
data_set = 'Hoosier_B1464run30';  % braking/traction (pure long. force) + combined
tyre_name = 'Hoosier';
% tyre geometric data:
% Hoosier	18.0x6.0-10
% 18 diameter in inches
% 6.0 section width in inches
% tread width in inches
diameter = 18*2.56;   % Converting inches to cm
Fz0 = 220;            % [N] nominal load is given
R0  = diameter/2/100; % [m] get from nominal load R0 (m)
to_rad = pi/180;
to_deg = 180/pi; 

fprintf('Loading dataset ...')
switch data_set
    case 'Hoosier_B1464run23'
        load ([data_set_path,data_set, '.mat']); % pure lateral
       cut_start = 32380;
       cut_end   = 54500;
    case 'Hoosier_B1464run30'
        load ([data_set_path,data_set,'.mat']); % pure longitudinal
        cut_start = 19028;
        cut_end   = 37643;
    otherwise
        error('Not found dataset: `%s`\n', data_set) ;
end

% select dataset portion
smpl_range = cut_start:cut_end;

fprintf('completed!\n')

%% Plot Raw Data

figure
tiledlayout(6,1)

ax_list(1) = nexttile; y_range = [min(min(-FZ),0) round(max(-FZ)*1.1)];
plot(-FZ)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')

ax_list(2) = nexttile; y_range = [min(min(IA),0) round(max(IA)*1.1)];
plot(IA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')
ylim('padded')

ax_list(3) = nexttile; y_range = [min(min(SA),0) round(max(SA)*1.1)];
plot(SA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Side slip')
xlabel('Samples [-]')
ylabel('[deg]')
ylim('padded')

ax_list(4) = nexttile; y_range = [min(min(SL),0) round(max(SL)*1.1)];
plot(SL)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Longitudinal slip')
xlabel('Samples [-]')
ylabel('[-]')

ax_list(5) = nexttile; y_range = [min(min(P),0) round(max(P)*1.1)];
plot(P)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre pressure')
xlabel('Samples [-]')
ylabel('[psi]')

ax_list(6) = nexttile;  y_range = [min(min(TSTC),0) round(max(TSTC)*1.1)];
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

linkaxes(ax_list,'x')

%% Select some specific data
% Cut useless data and select only 12 psi data

vec_samples = 1:1:length(smpl_range);
tyre_data = table(); % create empty table

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store raw data in the table
tyre_data.SL =  SL(smpl_range);                    % Slip Ratio based on RE (Longi.)
tyre_data.SA =  SA(smpl_range)*to_rad;             % Slip angle (Lateral)
tyre_data.FZ = -FZ(smpl_range);  % 0.453592  lb/kg % Vertical Load (- to get to ISO)
tyre_data.FX =  FX(smpl_range);                    % Longitudinal Force
tyre_data.FY =  FY(smpl_range);                    % Lateral Force (- to get to ISO)
tyre_data.MZ =  MZ(smpl_range);                    % Self Aliging Moment
tyre_data.IA =  IA(smpl_range)*to_rad;             % Inclination Angle (Camber)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
% Test data done at:
%  - 50lbf  ( 50*0.453592*9.81 =  223N )
%  - 150lbf (150*0.453592*9.81 =  667N )
%  - 200lbf (200*0.453592*9.81 =  890N )
%  - 250lbf (250*0.453592*9.81 = 1120N )

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

% The slip angle is varied continuously between -4 and +12°
% The slip angle is varied step wise for longitudinal slip tests
% 0° , - 3° , -6 °

SA_tol = 0.5*to_rad;
idx.SA_0    =  0-SA_tol          < tyre_data.SA & tyre_data.SA < 0+SA_tol;
idx.SA_3neg = -(3*to_rad+SA_tol) < tyre_data.SA & tyre_data.SA < -3*to_rad+SA_tol;
idx.SA_6neg = -(6*to_rad+SA_tol) < tyre_data.SA & tyre_data.SA < -6*to_rad+SA_tol;
SA_0     = tyre_data( idx.SA_0, : );
SA_3neg  = tyre_data( idx.SA_3neg, : );
SA_6neg  = tyre_data( idx.SA_6neg, : );

figure()
tiledlayout(3,1)

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


ax_list(3) = nexttile;
plot(tyre_data.SA*to_deg)
hold on
plot(vec_samples(idx.SA_0),   SA_0.SA*to_deg,'.');
plot(vec_samples(idx.SA_3neg),SA_3neg.SA*to_deg,'.');
plot(vec_samples(idx.SA_6neg),SA_6neg.SA*to_deg,'.');
title('Slide slip')
xlabel('Samples [-]')
ylabel('[rad]')

%% FX0 - PURE SLIP LONGITUDINAL FORCE
% Fitting with Fz=Fz_nom= 220 N, camber=0,  alpha = 0

[TDataTmp, ~] = intersect_table_data( SA_0, GAMMA_0, FZ_220 );
% extract data for zero slip and camber, and 220N

SL_vec = min(TDataTmp.SL):1e-4:max(TDataTmp.SL);

% Plot_selected_data
figure('Name','Selected-data')
plot_selected_data(TDataTmp);

FZ0 = mean(TDataTmp.FZ);

% Check guess
% zeros_vec = zeros(size(TDataTmp.SL));
% ones_vec  = ones(size(TDataTmp.SL));
% FX0_guess = MF96_FX0_vec(TDataTmp.SL, zeros_vec , zeros_vec, tyre_coeffs.FZ0*ones_vec, tyre_coeffs);
% 
% figure('Name', 'Guess')
% plot(TDataTmp.SL,TDataTmp.FX,'.', 'DisplayName', 'Raw')
% hold on
% plot(TDataTmp.SL, FX0_guess, '-', 'DisplayName', 'Guess')
% title('Pure longitudinal slip at $F_{z} = 220 N$, Camber = 0, Slip angle = 0 (Guess)')
% xlabel('$\kappa$ [-]')
% ylabel('$F_{x0}$ [N]')
% legend('Location','southeast')

% Plot raw data and initial guess
% figure()
% plot(TDataSub.KAPPA,TDataSub.FX,'o')
% hold on
% plot(TDataSub.KAPPA,FX0_guess,'x')

% Guess values for parameters to be optimised
% NOTE: many local minima => limits on parameters are fundamentals
% Limits for parameters to be optimised {lb: lower_bound, up: upper_bound}
% 1< pCx1 < 2
% 0< pEx1 < 1
%    [pCx1 pDx1 pEx1 pEx4  pHx1  pKx1  pVx1
P0 = [  1,   2,     1,   0,   0,     1,    0 ];
lb = [  1,   0.1,   0,   0,  -10,    0,   -10];
ub = [  2,    4,    1,   1,   10,    100,  10];

KAPPA_vec = TDataTmp.SL;
FX_vec    = TDataTmp.FX;

% LSM_pure_Fx returns the residual, so minimize the residual varying X. It
% is an unconstrained minimization problem

[P_fz_nom,~,~] = fmincon(@(P)resid_pure_Fx(P,FX_vec, KAPPA_vec,0,FZ0, tyre_coeffs),...
    P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values
tyre_coeffs.pCx1 = P_fz_nom(1) ; 
tyre_coeffs.pDx1 = P_fz_nom(2) ;
tyre_coeffs.pEx1 = P_fz_nom(3) ;
tyre_coeffs.pEx4 = P_fz_nom(4) ;
tyre_coeffs.pHx1 = P_fz_nom(5) ;
tyre_coeffs.pKx1 = P_fz_nom(6) ;
tyre_coeffs.pVx1 = P_fz_nom(7) ;

FX0_fz_nom_vec = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)) , zeros(size(SL_vec)), ...
    FZ0.*ones(size(SL_vec)),tyre_coeffs);

f = figure('Name','Fx0(Fz0)');
plot(TDataTmp.SL,TDataTmp.FX,'.','DisplayName', 'Raw')
hold on
plot(SL_vec,FX0_fz_nom_vec,'-','LineWidth',2,'DisplayName', 'Fitted')
title('Pure longitudinal slip at $F_{z} = 220 N$, Camber = 0, Slip angle = 0')
xlabel('$\kappa$ [-]')
ylabel('$F_{x0}$ [N]')
xlim([-0.25, 0.21]);
ylim([-850, 800]);
legend('Location','southeast')
exportgraphics(f,'GraphsISO/Fx0pure.eps')

res_Fx0  = resid_pure_Fx(P,FX_vec, KAPPA_vec,0,FZ0, tyre_coeffs);
R2 = 1-res_Fx0;
RMSE = sqrt(res_Fx0*sum(FX_vec.^2)/length(KAPPA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

%% FX0_dfz - LONGITUDINAL FORCE with VARIABLE LOAD FZ
% Extract data for zero slip and camber, and Variable load
[TDataTmp, ~] = intersect_table_data( SA_0, GAMMA_0 );

KAPPA_vec = TDataTmp.SL;
FX_vec    = TDataTmp.FX;
FZ_vec    = TDataTmp.FZ;

zeros_vec = zeros(size(TDataTmp.SL));

FX0_guess = MF96_FX0_vec(KAPPA_vec, zeros_vec , zeros_vec, FZ_vec, tyre_coeffs);

SL_vec = min(KAPPA_vec):1e-4:max(KAPPA_vec);
tmp_zeros = zeros(size(SL_vec));
tmp_ones = ones(size(SL_vec));

% FX0_fz_var_vec1 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
% FX0_fz_var_vec2 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
% FX0_fz_var_vec3 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
% FX0_fz_var_vec4 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

% Check guess
% figure('Name', 'Guess')
% plot(TDataTmp.SL,TDataTmp.FX,'.', 'DisplayName', 'Raw')
% hold on
% plot(SL_vec,FX0_fz_var_vec1,'-', 'DisplayName', '$F_{z} = 220N$(guess)')
% plot(SL_vec,FX0_fz_var_vec2,'-', 'DisplayName', '$F_{z} = 700N$(guess)')
% plot(SL_vec,FX0_fz_var_vec3,'-', 'DisplayName', '$F_{z} = 900N$(guess)')
% plot(SL_vec,FX0_fz_var_vec4,'-', 'DisplayName', '$F_{z} = 1120N$(guess)')
% title('Pure longitudinal slip at different vertical loads (Guess)')
% xlabel('$\kappa$ [-]')
% ylabel('$F_{x0}$ [N]')
% legend('Location','southeast')

% Guess values for parameters to be optimised
%    [pDx2 pEx2 pEx3 pHx2 pKx2 pKx3 pVx2]
P0 = [ -0.0255967912425912,-0.361847644865266,0.105853489495470,0.00105012690073064,-0.00191805758534771,0.153553880905060,-0.0255967912425912];
lb = [-1,  -1,  -1,  -1,  -1,  -1,  -1];
ub = [ 1,   1,   1,   1,   1,   1,   1];

% LSM_pure_Fx returns the residual, so minimize the residual varying X. It
% is an unconstrained minimization problem

[P_dfz,~,~] = fmincon(@(P)resid_pure_Fx_varFz(P,FX_vec, KAPPA_vec,0,FZ_vec, tyre_coeffs),...
    P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values
tyre_coeffs.pDx2 = P_dfz(1) ; % 1
tyre_coeffs.pEx2 = P_dfz(2) ;
tyre_coeffs.pEx3 = P_dfz(3) ;
tyre_coeffs.pHx2 = P_dfz(4) ;
tyre_coeffs.pKx2 = P_dfz(5) ;
tyre_coeffs.pKx3 = P_dfz(6) ;
tyre_coeffs.pVx2 = P_dfz(7) ;


res_FX0_dfz_vec = resid_pure_Fx_varFz(P_dfz,FX_vec,KAPPA_vec,0 , FZ_vec,tyre_coeffs);

FX0_fz_var_vec1 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec2 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec3 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec4 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);


f = figure('Name','Fx0(Fz)');
plot(TDataTmp.SL,TDataTmp.FX,'.','DisplayName','Raw')
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
%plot(SL_vec,FX0_dfz_vec,'-','LineWidth',2)
plot(SL_vec,FX0_fz_var_vec1,'-', 'LineWidth',2,'DisplayName','$F_{z} = 220N$')
plot(SL_vec,FX0_fz_var_vec2,'-', 'LineWidth',2,'DisplayName','$F_{z} = 700N$')
plot(SL_vec,FX0_fz_var_vec3,'-', 'LineWidth',2,'DisplayName','$F_{z} = 900N$')
plot(SL_vec,FX0_fz_var_vec4,'-', 'LineWidth',2,'DisplayName','$F_{z} = 1120N$')
title('Pure longitudinal slip at different vertical loads')
xlabel('$k$ [-]')
ylabel('$F_{x0}$ [N]')
legend('Location','southeast')
exportgraphics(f,'GraphsISO/Fx0dFz.eps')

R2 = 1-res_FX0_dfz_vec;
RMSE = sqrt(res_FX0_dfz_vec*sum(FX_vec.^2)/length(KAPPA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];
%% Cx_kappa - Cornering Stiffness
[kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_220.FZ), tyre_coeffs);
Calfa_vec1_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
[kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_700.FZ), tyre_coeffs);
Calfa_vec2_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
[kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_900.FZ), tyre_coeffs);
Calfa_vec3_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
[kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_1120.FZ), tyre_coeffs);
Calfa_vec4_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);

Calfa_vec1 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec2 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec3 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
Calfa_vec4 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

f = figure('Name','C_alpha');
subplot(2,1,1)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(mean(FZ_220.FZ),Calfa_vec1_0,'+','LineWidth',2)
plot(mean(FZ_700.FZ),Calfa_vec3_0,'+','LineWidth',2)
plot(mean(FZ_900.FZ),Calfa_vec4_0,'+','LineWidth',2)
plot(mean(FZ_1120.FZ),Calfa_vec2_0,'+','LineWidth',2)
xlabel('$F_z (N)$')
ylabel('$C_{k} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

subplot(2,1,2)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(SL_vec,Calfa_vec1,'-','LineWidth',2)
plot(SL_vec,Calfa_vec2,'-','LineWidth',2)
plot(SL_vec,Calfa_vec3,'-','LineWidth',2)
plot(SL_vec,Calfa_vec4,'-','LineWidth',2)
xlabel('$k (rad)$')
ylabel('$C_{k} (N/rad)$')
legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
exportgraphics(f,'GraphsISO/C_kappa.eps')

%% FX0_gamma - LONGITUDINAL FORCE with VARIABLE CAMBER
% Extract data with no slip and nominal load
[TDataTmp, ~] = intersect_table_data( SA_0, FZ_220 );

% Fit the coeff {pDx3}
zeros_vec = zeros(size(TDataTmp.SL));
ones_vec  = ones(size(TDataTmp.SL));

% Guess values for parameters to be optimised
%  [pDx3]
P0 = 10.8761875333065;

% NOTE: many local minima => limits on parameters are fundamentals
% Limits for parameters to be optimised {lb: lower_bound, up: upper_bound}
lb = [];
ub = [];

KAPPA_vec = TDataTmp.SL;
GAMMA_vec = TDataTmp.IA;
FX_vec    = TDataTmp.FX;
FZ_vec    = TDataTmp.FZ;

figure()
plot(KAPPA_vec,FX_vec);
xlabel('$\kappa$')
ylabel('$F_{x}$ [N]')


% LSM_pure_Fx returns the residual, so minimize the residual varying X. It
% is an unconstrained minimization problem

[P_varGamma,fval,exitflag] = fmincon(@(P)resid_pure_Fx_varGamma(P,FX_vec, KAPPA_vec,GAMMA_vec,tyre_coeffs.FZ0, tyre_coeffs),...
    P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values
tyre_coeffs.pDx3 = P_varGamma(1) ; % 1

FX0_varGamma_vec0 = MF96_FX0_vec(KAPPA_vec, zeros_vec , GAMMA_0.IA, tyre_coeffs.FZ0*ones_vec,tyre_coeffs);
FX0_varGamma_vec2 = MF96_FX0_vec(KAPPA_vec, zeros_vec , GAMMA_2.IA, tyre_coeffs.FZ0*ones_vec,tyre_coeffs);
FX0_varGamma_vec4 = MF96_FX0_vec(KAPPA_vec, zeros_vec , GAMMA_4.IA, tyre_coeffs.FZ0*ones_vec,tyre_coeffs);

f = figure('Name','Fx0(Gamma)');
plot(KAPPA_vec,TDataTmp.FX,'.', 'DisplayName','Raw')
hold on
plot(KAPPA_vec,FX0_varGamma_vec0,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 0$')
plot(KAPPA_vec,FX0_varGamma_vec2,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 2$')
plot(KAPPA_vec,FX0_varGamma_vec4,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 4$')
%title('Pure Longitudinal Slip Force with Variable Camber ')
xlabel('$\kappa$ [-]')
ylabel('$F_{x0}$ [N]')
legend('Location','southeast')
exportgraphics(f,'GraphsISO/Fx0_gamma.eps')

% Calculate the residuals with the optimal solution found above
res_Fx0_varGamma  = resid_pure_Fx_varGamma(P_varGamma,FX_vec, KAPPA_vec,GAMMA_vec,tyre_coeffs.FZ0, tyre_coeffs);
R2 = 1-res_Fx0_varGamma;
RMSE = sqrt(res_Fx0_varGamma*sum(FX_vec.^2)/length(KAPPA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];


% [kappa__x, Bx, Cx, Dx, Ex, SVx] = MF96_FX0_coeffs(0, 0, GAMMA_vec(3), tyre_coeffs.FZ0, tyre_coeffs);
% %
% fprintf('Bx      = %6.3f\n',Bx);
% fprintf('Cx      = %6.3f\n',Cx);
% fprintf('mux      = %6.3f\n',Dx/tyre_coeffs.FZ0);
% fprintf('Ex      = %6.3f\n',Ex);
% fprintf('SVx     = %6.3f\n',SVx);
% fprintf('kappa_x = %6.3f\n',kappa__x);
% fprintf('Kx      = %6.3f\n',Bx*Cx*Dx/tyre_coeffs.FZ0);

% Longitudinal stiffness
% Kx_vec = zeros(size(load_vec));
% for i = 1:length(load_vec)
%   [kappa__x, Bx, Cx, Dx, Ex, SVx] = MF96_FX0_coeffs(0, 0, 0, load_vec(i), tyre_coeffs);
%   Kx_vec(i) = Bx*Cx*Dx/tyre_data.Fz0;
% end
%
% figure('Name','Kx vs Fz')
% plot(load_vec,Kx_vec,'o-')

%% FX - Combined Slip Longitudinal Force !!FIX WEIGHTS!!
% Fz=220N, variable alpha, p=12psi, gamma=0
% Fit Longitudal Force with Combined Slip for VARIABLE slip angle ALPHA
[TDataTmp, ~] = intersect_table_data(GAMMA_0, FZ_220);

%Plotting raw data
figure('Name','Raw Data')
plot(TDataTmp.SL,TDataTmp.FX);

% Initialise values for parameters to be optimised
%[rBx1, rBx2, rCx1, rHx1]
P0 = [8.3, 5, 1, 0];
lb = [];
ub = [];

KAPPA_vec = TDataTmp.SL; % extract for clarity
ALPHA_vec = TDataTmp.SA;
FZ0 = mean(TDataTmp.FZ);

[P_comb,~,~] = fmincon(@(P)resid_comb_Fx(P,TDataTmp.FX,KAPPA_vec,ALPHA_vec,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% Change tyre data with new optimal values                             
tyre_coeffs.rBx1 = P_comb(1) ; 
tyre_coeffs.rBx2 = P_comb(2) ;  
tyre_coeffs.rCx1 = P_comb(3) ;
tyre_coeffs.rHx1 = P_comb(4) ;

fx_SA0_vec = MF96_FXcomb_vect(KAPPA_vec, mean(SA_0.SA).*ones(size(KAPPA_vec)),    zeros(size(KAPPA_vec)), FZ0.*ones(size(KAPPA_vec)), tyre_coeffs);
fx_SA3_vec = MF96_FXcomb_vect(KAPPA_vec, mean(SA_3neg.SA).*ones(size(KAPPA_vec)), zeros(size(KAPPA_vec)), FZ0.*ones(size(KAPPA_vec)), tyre_coeffs);
fx_SA6_vec = MF96_FXcomb_vect(KAPPA_vec, mean(SA_6neg.SA).*ones(size(KAPPA_vec)), zeros(size(KAPPA_vec)), FZ0.*ones(size(KAPPA_vec)), tyre_coeffs);


f = figure('Name','Fx(k)');
hold on
plot(KAPPA_vec,TDataTmp.FX,'.')
plot(KAPPA_vec,fx_SA0_vec,'r','LineWidth',1.5)
plot(KAPPA_vec,fx_SA3_vec,'g','LineWidth',1.5)
plot(KAPPA_vec,fx_SA6_vec,'b','LineWidth',1.5)
xlabel('long. slip $k(-)$')
ylabel('$F_x(N)$')
ylim('padded')
legend('Raw Data','Fitted $\alpha$=0 ',...
    'Fitted $\alpha$=-3 deg','Fitted $\alpha$=-6 deg',Location='southeast')
title('Combined Slip Longitudinal Force')
exportgraphics(f,'GraphsISO/Fx_Comb.eps')

% Error
res_Fx  = resid_comb_Fx(P_comb,TDataTmp.FX,KAPPA_vec,ALPHA_vec,FZ0,tyre_coeffs);
R2 = 1-res_Fx;
RMSE = sqrt(res_Fx*sum(TDataTmp.FX.^2)/length(KAPPA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

% Weights
Gxa = zeros(length(KAPPA_vec),3);
SL_vec = min(KAPPA_vec):1e-4:max(KAPPA_vec);
for i=1:length(SL_vec)
    [Gxa(i,1),~,~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_0.SA),    0, FZ0, tyre_coeffs);
    [Gxa(i,2),~,~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_3neg.SA), 0, FZ0, tyre_coeffs);
    [Gxa(i,3),~,~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_6neg.SA), 0, FZ0, tyre_coeffs);
end
figure('Name','Gxa(k)')
plot(SL_vec,Gxa)
xlabel('k')
ylabel('Gxa')
legend(['$\alpha$',' = ',num2str(0),' deg'],['$\alpha$',' = ',num2str(3),...
    ' deg'],['$\alpha$',' = ',num2str(6),' deg'],'Location','best')
title('Gxa(k)')

%% FY - Combined Slip Lateral Force 
[TDataTmp, ~] = intersect_table_data(GAMMA_0, FZ_220);

FY_vec    = TDataTmp.FY;
ALPHA_vec = TDataTmp.SA; % steps
KAPPA_vec = TDataTmp.SL; % sweeps
FZ0 = mean(FZ_220.FZ);

% Fit Coefficients
%    [rBy1,rBy2,rBy3,rCy1,rHy1,rVy1,rVy4,rVy5,rVy6]
P0 = [4.9,2.2,0,1,0.01,0.01,50,1,20];
% P0 = [2,3,0.002,2,0.04,-0.2,1,-0.2,-0.2]; %same
lb = [];
ub = [];

[P_comb,~,~] = fmincon(@(P)resid_comb_Fy(P,FY_vec,KAPPA_vec,ALPHA_vec,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

tyre_coeffs.rBy1 = P_comb(1) ;
tyre_coeffs.rBy2 = P_comb(2) ;
tyre_coeffs.rBy3 = P_comb(3) ;
tyre_coeffs.rCy1 = P_comb(4) ;
tyre_coeffs.rHy1 = P_comb(5) ;
tyre_coeffs.rVy1 = P_comb(6) ;
tyre_coeffs.rVy4 = P_comb(7) ;
tyre_coeffs.rVy5 = P_comb(8) ;
tyre_coeffs.rVy6 = P_comb(9) ;

SL_vec = min(KAPPA_vec):1e-4:max(KAPPA_vec);
ones_vec  = ones(size(SL_vec));
zeros_vec = zeros(size(SL_vec));

fy_vec0 = MF96_FYcomb_vect(SL_vec, mean(SA_0.SA).*ones_vec,    zeros_vec, FZ0.*ones_vec, tyre_coeffs);
fy_vec3 = MF96_FYcomb_vect(SL_vec, mean(SA_3neg.SA).*ones_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);
fy_vec6 = MF96_FYcomb_vect(SL_vec, mean(SA_6neg.SA).*ones_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);

% Plot Raw and Fitted Data
f = figure;
set(groot,'defaulttextinterpreter','latex') 
hold on
plot(KAPPA_vec,FY_vec,'b.')
plot(SL_vec,fy_vec0,'r','LineWidth',1.5);
plot(SL_vec,fy_vec3,'g','LineWidth',1.5);
plot(SL_vec,fy_vec6,'c','LineWidth',1.5);
xlabel('$k(-)$ ')
ylabel('$F_y(N)$')
legend('Raw Data',['$\alpha$',' = ',num2str(0)],['$\alpha$',' = ',num2str(3)],['$\alpha$',' = ',num2str(6)],Location='best');
title('Combined Slip Lateral Force')
exportgraphics(f,'GraphsISO/Fy_comb.eps');

res_Fy  = resid_comb_Fy(P,FY_vec,KAPPA_vec,ALPHA_vec,FZ0,tyre_coeffs);
R2 = 1-res_Fy;
RMSE = sqrt(res_Fy*sum(FY_vec.^2)/length(KAPPA_vec));
fprintf('R^2 = %6.3f \nRMSE = %6.3f \n', R2, RMSE );
err = [err ; R2 RMSE];

Gyk = zeros(length(KAPPA_vec),3);
for i=1:length(SL_vec)
    [~,Gyk(i,1),~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_0.SA), 0, FZ0, tyre_coeffs);
    [~,Gyk(i,2),~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_3neg.SA), 0, FZ0, tyre_coeffs);
    [~,Gyk(i,3),~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_6neg.SA), 0, FZ0, tyre_coeffs);
end
figure('Name','Gyk(k)')
plot(SL_vec,Gyk)
xlabel('k')
ylabel('Gyk')
title('Gyk(k)')

%% Save tyre data structure to mat file
save(['tyre_' tyre_name,'.mat'],'tyre_coeffs');
TTT = rows2vars(struct2table(tyre_coeffs));
delete ttt.xls
writetable(TTT,'ttt.xls')
T_err = table(err(:,1),err(:,2),'VariableNames',["R2","RMSE"]);
delete t_err.xls
writetable(T_err,'t_err.xls')

%% Extras
% sweep k and use alpha steps
SL_vec = -15*to_rad:0.001:15*to_rad;
SA_vec = (-6:1:6)*to_rad;
ones_vec = ones(size(SL_vec));
zeros_vec = zeros(size(SL_vec));
FZ0 = 220;

fx_vec = zeros(length(SL_vec),length(SA_vec));
fy_vec = zeros(length(SL_vec),length(SA_vec));

for i=1:length(SA_vec)
    fx_vec(:,i) = MF96_FXcomb_vect(SL_vec, SA_vec(i).*ones_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);
    fy_vec(:,i) = MF96_FYcomb_vect(SL_vec, SA_vec(i).*ones_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);
end


% figure('Name','Friction Ellipse')
% plot(fy_vec,fx_vec)



