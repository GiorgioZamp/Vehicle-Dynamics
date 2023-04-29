%  lsqcurvefit
tmp = intersect_table_data(GAMMA_0,P_80);

%remove all row with 0 Fz
toDelete = tmp.FZ == 0;
tmp(toDelete,:) = [];
tmp;

alpha0 = tmp.SA;
mz0 = tmp.MZ;
fz0 = tmp.FZ;
tmp_tyre_data = tyre_coeffs;

P0 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1];

fun = @(P,alpha) MF96_Mz0_MOD(P,zeros(size(mz0)), alpha0, zeros(size(mz0)), fz0, tmp_tyre_data);
P_mzdfz(1,:) = lsqcurvefit(fun,P0,alpha0,mz0);

tmp_tyre_data.qHz2  = P_mzdfz(1) ;
tmp_tyre_data.qBz2  = P_mzdfz(2) ;
tmp_tyre_data.qBz3  = P_mzdfz(3) ;
tmp_tyre_data.qDz2  = P_mzdfz(4) ;
tmp_tyre_data.qEz2  = P_mzdfz(5) ;
tmp_tyre_data.qEz3  = P_mzdfz(6) ;
tmp_tyre_data.qDz7  = P_mzdfz(7) ;

SA_vec = min(alpha0):0.001:max(alpha0); % side slip vector [rad]
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