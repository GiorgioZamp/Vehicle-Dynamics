
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
lb = 0;
ub = 1;

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