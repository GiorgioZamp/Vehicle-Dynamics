[TDataDFz, ~] = intersect_table_data(FZ_440, GAMMA_0);
ALPHA_vec = TDataDFz.SA;
FY_vec    = TDataDFz.FY;
FZ_vec    = TDataDFz.FZ;
SA_vec = min(ALPHA_vec):0.001:max(ALPHA_vec);
FZ0 = mean(FZ_vec);

P0 = [-0.1,-0.1,-0.1,-0.1,-0.1];
% P_dfz = fminunc(@(P)resid_pure_Fy_varFz(P,FY_vec,ALPHA_vec,0,FZ_vec,tyre_coeffs),P0);
fun = @(P,alpha) MF96_FY0_MOD(P,zeros(size(ALPHA_vec)), alpha, zeros(size(ALPHA_vec)), FZ_vec, tyre_coeffs);
P_dfz = lsqcurvefit(fun,P0,ALPHA_vec,FY_vec);

tyre_coeffs.pDy2 = P_dfz(1);
tyre_coeffs.pEy2 = P_dfz(2);
tyre_coeffs.pEy3 = P_dfz(3);
tyre_coeffs.pHy2 = P_dfz(4);
tyre_coeffs.pVy2 = P_dfz(5);

FY0_fz_var_vec = MF96_FY0_vec(zeros(size(SA_vec)), SA_vec, zeros(size(SA_vec)), ...
                              FZ0.*ones(size(SA_vec)),tyre_coeffs);

figure,hold on;
plot(ALPHA_vec*to_deg,FY_vec,'.')
plot(SA_vec*to_deg,FY0_fz_var_vec,'-','LineWidth',2)