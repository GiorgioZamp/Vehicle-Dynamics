function [Mz0_fit] = fit_Mz_0(GAMMA_0,FZ_220,tyre_coeffs)

% Fit the self-aligning moment

[TDataMz, ~] = intersect_table_data( GAMMA_0, FZ_220 );

ALPHA_vec = TDataMz.SA;
MZ_vec    = TDataMz.MZ;

FZ0 = mean(TDataMz.FZ);

% Guess values for parameters to be optimised
%    [qBz1,qBz9,qBz10,qCz1,qDz1,qDz6,qEz1,qEz4,qHz1]
P0 = [ 1,   1,   1,    1,   1,   1,   1,   1,   1  ];
lb = [  ];
ub = [  ];

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

SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

[alpha__y, By, Cy, Dy, Ey, SHy, SVy, ~] = MF96_FXFYCOMB_coeffs(0, ALPHA_vec, 0, FZ0, tyre_coeffs);
Fy = magic_formula(alpha__y, By, Cy, Dy, Ey, SVy);

Mz0_fit = MF96_Mz0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)), ...
                        FZ0.*ones(size(SA_vec)), Fy, SHy, SVy, tyre_coeffs);


% Plot Raw Data and Fitted Function
figure('Name','Mz0(Fz0)')
plot(ALPHA_vec*to_deg,TData0.MZ,'.')
hold on
plot(SA_vec*to_deg,Mz0_fit,'-','LineWidth',2)
xlabel('$\alpha$ [deg]')
ylabel('$M_{z0}$ [Nm]')
legend('Raw','Fitted')

