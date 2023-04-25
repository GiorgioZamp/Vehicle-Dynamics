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

addpath('tyre_lib/','dataset\')

to_rad = pi/180;
to_deg = 180/pi;

%% Select tyre dataset

% dataset path
data_set_path = 'dataset/';

% dataset selection and loading
data_set = 'Hoosier_B1464run23'; % pure lateral forces
% data_set = 'Hoosier_B1464run30';  % braking/traction (pure log. force) + combined

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
        cut_start = 27760;
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
tiledlayout(7,1)

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

ax_list(7) = nexttile;  y_range = [min(min(FX),0) round(max(FX)*1.1)];
plot(FX,'DisplayName','Center')
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Longitudinal Force')
xlabel('Samples [-]')
ylabel('[N]')

linkaxes(ax_list,'x')

%% Select some specific data
% Cut crappy data and select only 12 psi data

vec_samples = 1:1:length(smpl_range);
tyre_data = table(); % create empty table

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store raw data in the table
tyre_data.SL =  SL(smpl_range);                    %Slip Ratio based on RE (Longi.)
tyre_data.SA =  SA(smpl_range)*to_rad;             %Slip angle (Lateral)
tyre_data.FZ = -FZ(smpl_range);  % 0.453592  lb/kg %Verticle Load
tyre_data.FX =  FX(smpl_range);                    %Longitudinal Force
tyre_data.FY = -FY(smpl_range);                    %Lateral Force
tyre_data.MZ =  MZ(smpl_range);                    %Self Aliging Moments
tyre_data.IA =  IA(smpl_range)*to_rad;             %Inclination Angle (Camber)
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

%% Intersect tables to obtain specific sub-datasets

[TData0, ~] = intersect_table_data( SA_0, GAMMA_0, FZ_220 );
% extract data for zero slip and camber, and 220N

SL_vec = min(TData0.SL):1e-4:max(TData0.SL);

% Plot_selected_data
figure('Name','Selected-data')
plot_selected_data(TData0);

% FITTING
% initialise tyre data
tyre_coeffs = initialise_tyre_data(R0, Fz0);
% Load coefficients from lateral test
% load('tyre_Hoosier_B1464run23.mat')

% Fitting with Fz=Fz_nom= 220N and camber=0  alpha = 0 VX= 10
% ------------------
% Logitudinal slip

% Fit the coeffs {pCx1, pDx1, pEx1, pEx4, pKx1, pHx1, pVx1}
FZ0 = mean(TData0.FZ);

zeros_vec = zeros(size(TData0.SL));
ones_vec  = ones(size(TData0.SL));

FX0_guess = MF96_FX0_vec(TData0.SL, zeros_vec , zeros_vec, tyre_coeffs.FZ0*ones_vec, tyre_coeffs);

% check guess
figure('Name', 'Guess')
plot(TData0.SL,TData0.FX,'.', 'DisplayName', 'Raw')
hold on
plot(TData0.SL, FX0_guess, '-', 'DisplayName', 'Guess')
title('Pure longitudinal slip at $F_{z} = 220 N$, Camber = 0, Slip angle = 0 (Guess)')
xlabel('$\kappa$ [-]')
ylabel('$F_{x0}$ [N]')
legend('Location','southeast')

% Plot raw data and initial guess
% figure()
% plot(TDataSub.KAPPA,TDataSub.FX,'o')
% hold on
% plot(TDataSub.KAPPA,FX0_guess,'x')

% Guess values for parameters to be optimised
%    [pCx1 pDx1 pEx1 pEx4  pHx1  pKx1  pVx1
P0 = [  1,   2,   1,  0,   0,   1,   0];

% NOTE: many local minima => limits on parameters are fundamentals
% Limits for parameters to be optimised {lb: lower_bound, up: upper_bound}
% 1< pCx1 < 2
% 0< pEx1 < 1
%    [pCx1 pDx1 pEx1 pEx4  pHx1  pKx1  pVx1
lb = [1,   0.1,   0,   0,  -10,    0,   -10];
ub = [2,    4,    1,   1,   10,    100,  10];

KAPPA_vec = TData0.SL;
FX_vec    = TData0.FX;

% check guess
% SL_vec = -0.3:0.001:0.3;
% FX0_fz_nom_vec = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)) , zeros(size(SL_vec)), ...
%     FZ0.*ones(size(SL_vec)),tyre_coeffs);
% figure
% plot(KAPPA_vec,FX_vec,'.')
% hold on
% plot(SL_vec,FX0_fz_nom_vec,'.')
%


% LSM_pure_Fx returns the residual, so minimize the residual varying X. It
% is an unconstrained minimization problem

[P_fz_nom,~,~] = fmincon(@(P)resid_pure_Fx(P,FX_vec, KAPPA_vec,0,FZ0, tyre_coeffs),...
    P0,[],[],[],[],lb,ub);

% Update tyre data with new optimal values
tyre_coeffs.pCx1 = P_fz_nom(1) ; % 1
tyre_coeffs.pDx1 = P_fz_nom(2) ;
tyre_coeffs.pEx1 = P_fz_nom(3) ;
tyre_coeffs.pEx4 = P_fz_nom(4) ;
tyre_coeffs.pHx1 = P_fz_nom(5) ;
tyre_coeffs.pKx1 = P_fz_nom(6) ;
tyre_coeffs.pVx1 = P_fz_nom(7) ;

FX0_fz_nom_vec = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)) , zeros(size(SL_vec)), ...
    FZ0.*ones(size(SL_vec)),tyre_coeffs);

figure('Name','Fx0(Fz0)')
plot(TData0.SL,TData0.FX,'.','DisplayName', 'Raw')
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(SL_vec,FX0_fz_nom_vec,'-','LineWidth',2,'DisplayName', 'Fitted')
title('Pure longitudinal slip at $F_{z} = 220 N$, Camber = 0, Slip angle = 0')
xlabel('$\kappa$ [-]')
ylabel('$F_{x0}$ [N]')
xlim([-0.25, 0.21]);
ylim([-850, 800]);
legend('Location','southeast')

%% Fit coeefficients with VARIABLE LOAD
% Extract data for zero slip and camber, and Variable load
[TDataDFz, ~] = intersect_table_data( SA_0, GAMMA_0 );

KAPPA_vec = TDataDFz.SL;
FX_vec    = TDataDFz.FX;
FZ_vec    = TDataDFz.FZ;

zeros_vec = zeros(size(TDataDFz.SL));

FX0_guess = MF96_FX0_vec(KAPPA_vec, zeros_vec , zeros_vec, FZ_vec, tyre_coeffs);

SL_vec = min(KAPPA_vec):1e-4:max(KAPPA_vec);
tmp_zeros = zeros(size(SL_vec));
tmp_ones = ones(size(SL_vec));

FX0_fz_var_vec1 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec2 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec3 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
FX0_fz_var_vec4 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

% Check guess
figure('Name', 'Guess')
plot(TDataDFz.SL,TDataDFz.FX,'.', 'DisplayName', 'Raw')
hold on
plot(SL_vec,FX0_fz_var_vec1,'-', 'DisplayName', '$F_{z} = 220N$(guess)')
plot(SL_vec,FX0_fz_var_vec2,'-', 'DisplayName', '$F_{z} = 700N$(guess)')
plot(SL_vec,FX0_fz_var_vec3,'-', 'DisplayName', '$F_{z} = 900N$(guess)')
plot(SL_vec,FX0_fz_var_vec4,'-', 'DisplayName', '$F_{z} = 1120N$(guess)')
title('Pure longitudinal slip at different vertical loads (Guess)')
xlabel('$\kappa$ [-]')
ylabel('$F_{x0}$ [N]')
legend('Location','southeast')

% Guess values for parameters to be optimised
%    [pDx2 pEx2 pEx3 pHx2 pKx2 pKx3 pVx2]
P0 = [ 0,   0,   0,   0,   0,   0,   0];
lb = [];
ub = [];


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


figure('Name','Fx0(Fz)')
plot(TDataDFz.SL,TDataDFz.FX,'.','DisplayName','Raw')
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
%plot(SL_vec,FX0_dfz_vec,'-','LineWidth',2)
plot(SL_vec,FX0_fz_var_vec1,'-', 'LineWidth',2,'DisplayName','$F_{z} = 220N$')
plot(SL_vec,FX0_fz_var_vec2,'-', 'LineWidth',2,'DisplayName','$F_{z} = 700N$')
plot(SL_vec,FX0_fz_var_vec3,'-', 'LineWidth',2,'DisplayName','$F_{z} = 900N$')
plot(SL_vec,FX0_fz_var_vec4,'-', 'LineWidth',2,'DisplayName','$F_{z} = 1120N$')
title('Pure longitudinal slip at different vertical loads')
ylabel('$F_{x0}$ [N]')
legend('Location','southeast')

%% Cornering Stiffness
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

figure('Name','C_alpha')
subplot(2,1,1)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(mean(FZ_220.FZ),Calfa_vec1_0,'+','LineWidth',2)
plot(mean(FZ_700.FZ),Calfa_vec3_0,'+','LineWidth',2)
plot(mean(FZ_900.FZ),Calfa_vec4_0,'+','LineWidth',2)
plot(mean(FZ_1120.FZ),Calfa_vec2_0,'+','LineWidth',2)
legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

subplot(2,1,2)
hold on
%plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
plot(SL_vec,Calfa_vec1,'-','LineWidth',2)
plot(SL_vec,Calfa_vec2,'-','LineWidth',2)
plot(SL_vec,Calfa_vec3,'-','LineWidth',2)
plot(SL_vec,Calfa_vec4,'-','LineWidth',2)
legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})

%% Fit coefficient with VARIABLE CAMBER
% Extract data with no slip and nominal load
[TDataGamma, ~] = intersect_table_data( SA_0, FZ_220 );

% Fit the coeff {pDx3}
zeros_vec = zeros(size(TDataGamma.SL));
ones_vec  = ones(size(TDataGamma.SL));

% Guess values for parameters to be optimised
%  [pDx3]
P0 = 0;

% NOTE: many local minima => limits on parameters are fundamentals
% Limits for parameters to be optimised {lb: lower_bound, up: upper_bound}
% 1< pCx1 < 2
% 0< pEx1 < 1
%lb = [0, 0,  0, 0,  0,  0,  0];
%ub = [2, 1e6,1, 1,1e1,1e2,1e2];
lb = [];
ub = [];

KAPPA_vec = TDataGamma.SL;
GAMMA_vec = TDataGamma.IA;
FX_vec    = TDataGamma.FX;
FZ_vec    = TDataGamma.FZ;

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

figure('Name','Fx0(Gamma)')
plot(KAPPA_vec,TDataGamma.FX,'.', 'DisplayName','Raw')
hold on
plot(KAPPA_vec,FX0_varGamma_vec0,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 0$')
plot(KAPPA_vec,FX0_varGamma_vec2,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 2$')
plot(KAPPA_vec,FX0_varGamma_vec4,'-', 'LineWidth',1, 'DisplayName',  'Fitted $\gamma = 4$')
title('Pure Longitudinal slip at verticle load $F_{z}$ = 220[N], $\alpha = 0$ ')
xlabel('$\kappa$ [-]')
ylabel('$F_{x}$ [N]')
legend('Location','southeast')

% Calculate the residuals with the optimal solution found above
res_Fx0_varGamma  = resid_pure_Fx_varGamma(P_varGamma,FX_vec, KAPPA_vec,GAMMA_vec,tyre_coeffs.FZ0, tyre_coeffs);

% R-squared is
% 1-SSE/SST
% SSE/SST = res_Fx0_nom

% SSE is the sum of squared error,  SST is the sum of squared total
fprintf('R-squared = %6.3f\n',1-res_Fx0_varGamma);


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


%% Combined Slip Longitudinal Force
% Fz=220N, variable alpha, p=12psi, gamma=0
% Fit Longitudal Force with Combined Slip for VARIABLE slip angle ALPHA
[TDataComb, ~] = intersect_table_data(GAMMA_0, FZ_220);

%Plotting raw data
figure('Name','Raw Data')
plot(TDataComb.SL,TDataComb.FX);

% Initialise values for parameters to be optimised
%[rBx1, rBx2, rCx1, rHx1]
P0 = [10, 5, 1, 0];
lb = [1];
ub = [10];

KAPPAComb_vec = TDataComb.SL; % extract for clarity
ALPHA_vec = TDataComb.SA;

FX_vec =  MF96_FX0_vec(KAPPAComb_vec, zeros(size(KAPPAComb_vec)), zeros(size(KAPPAComb_vec)), FZ0.*ones(size(KAPPAComb_vec)), tyre_coeffs);

[P_comb,~,~] = fmincon(@(P)resid_comb_Fx(P,FX_vec,TDataComb.FX,KAPPAComb_vec,ALPHA_vec,FZ0,tyre_coeffs),...
                               P0,[],[],[],[],lb,ub);

% Change tyre data with new optimal values                             
tyre_coeffs.rBx1 = P_comb(1) ; 
tyre_coeffs.rBx2 = P_comb(2) ;  
tyre_coeffs.rCx1 = P_comb(3) ;
tyre_coeffs.rHx1 = P_comb(4) ;

fx_SA0_vec = MF96_FXcomb_vect(FX_vec, KAPPAComb_vec, mean(SA_0.SA).*ones(size(KAPPAComb_vec)), zeros(size(KAPPAComb_vec)), FZ0.*ones(size(KAPPAComb_vec)), tyre_coeffs);
fx_SA3_vec = MF96_FXcomb_vect(FX_vec, KAPPAComb_vec, mean(SA_3neg.SA).*ones(size(KAPPAComb_vec)), zeros(size(KAPPAComb_vec)), FZ0.*ones(size(KAPPAComb_vec)), tyre_coeffs);
fx_SA6_vec = MF96_FXcomb_vect(FX_vec, KAPPAComb_vec, mean(SA_6neg.SA).*ones(size(KAPPAComb_vec)), zeros(size(KAPPAComb_vec)), FZ0.*ones(size(KAPPAComb_vec)), tyre_coeffs);


figure, grid on, hold on;
plot(KAPPAComb_vec*to_deg,TDataComb.FX,'.')
plot(KAPPAComb_vec*to_deg,fx_SA0_vec,'r','LineWidth',1.5)
plot(KAPPAComb_vec*to_deg,fx_SA3_vec,'g','LineWidth',1.5)
plot(KAPPAComb_vec*to_deg,fx_SA6_vec,'b','LineWidth',1.5)
xlabel('long. slip $k(-)$')
ylabel('$F_x(N)$')
ylim('padded')
legend('Raw Data','Fitted $\alpha$=0 deg',...
    'Fitted $\alpha$=-3 deg','Fitted $\alpha$=-6 deg',Location='southeast')
title('Combined Slip Longitudinal Force')
%% Plot Weights G Behaviour
%
% % Compute Gxa(k)
% sa = [0,3,6,10,20]; % side slip in radians
% sl = linspace(-1,1,1e4);   % longitudinal slip
%
% Gxa_k = zeros(length(sa),length(sl));
% for i = 1:length(sa)
%     for j = 1:length(sl)
%         Gxa_k(i,j) = MF96_FXFYCOMB_coeffs(sl(j), sa(i)*to_rad, 0, FZ0, tyre_coeffs); %alpha row, k column
%     end
% end
%
% % Plot Gxa(k)
% figure, grid on, hold on;
% plot(sl,Gxa_k)
% xlabel('longitudinal slip $k$(-)')
% ylabel('$G_{xa}(-)$')
% ylim('padded')
% leg = cell(size(sa));
% for i = 1:length(sa)
%     leg{i} = ['$\alpha$ = ',num2str(sa(i)),' deg'];
% end
% legend(leg,Location="best")
% title('Weighting function $G_{xa}$ as a function of $k$')
% hold off
%
% % Compute Gxa(alpha)
% sa = linspace(-20,20,1e4);
% sl = [0,0.1,0.2,0.5,0.8,1];
% Gxa_a = zeros(length(sl),length(sa));
% for i = 1:length(sl)
%     for j = 1:length(sa)
%         Gxa_a(i,j) = MF96_FXFYCOMB_coeffs(sl(i), sa(j), 0, FZ0, tyre_coeffs); % k row, alpha column
%     end
% end
%
% % Plot Gxa(alpha)
% figure, grid on, hold on;
% plot(sa,Gxa_a)
% xlabel('side slip angle $\alpha$(deg)')
% ylabel('$G_{xa}(-)$')
% ylim('padded')
% leg = cell(size(sl));
% for i = 1:length(sl)
%     leg{i} = ['$k$ = ',num2str(sl(i))];
% end
% legend(leg,Location="best")
% title('Weighting function $G_{xa}$ as a function of $\alpha$')
% hold off

%% Combined Slip Lateral Force FY
[TDataComb, ~] = intersect_table_data(GAMMA_0, FZ_220, SA_3neg);

FY_vec    = TDataComb.FY;
ALPHA_vec = TDataComb.SA;
KAPPA_vec = TDataComb.SL;
FZ_vec    = TDataComb.FZ;
ones_vec  = ones(size(ALPHA_vec));
zeros_vec = zeros(size(ALPHA_vec));
FZ0 = mean(FZ_vec);


% Fit Coefficients
%    [rBy1,rBy2,rBy3,rCy1,rHy1,rVy1,rVy4,rVy5,rVy6]
P0 = [2,3,0.002,2,0.04,-0.2,1,-0.2,-0.2];
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

SA_vec = min(ALPHA_vec):1e-4:max(ALPHA_vec);

fy_vec = MF96_FYcomb_vect(KAPPA_vec, ALPHA_vec, zeros_vec, FZ0.*ones_vec, tyre_coeffs);

% Plot Raw and Fitted Data
figure, grid on, hold on;
plot(ALPHA_vec*to_deg,FY_vec,'b.')
plot(ALPHA_vec,fy_vec);
xlabel('$\alpha(°)$ ')
ylabel('$F_y(N)$')
legend(leg,Location='best')
title('Combined Slip Lateral Force')

%% Save tyre data structure to mat file
% save(['tyre_' data_set,'.mat'],'tyre_coeffs');

