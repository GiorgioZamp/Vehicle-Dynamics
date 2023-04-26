% Self Aligning Moment Mz0
% this function remap the scalar function to its vectorial form
function [Mz0_vec] = MF96_Mz0_vec(kappa_vec, alpha_vec, phi_vec, Fz_vec, tyre_data)

Mz0_vec = zeros(size(alpha_vec));
for i = 1:length(alpha_vec)
    % precode
    fy0 = MF96_FY0(0, alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    [alpha__t, Bt, Ct, Dt, Et, Br, Dr, alpha__r] = MF96_MZ0_coeffs(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    % main code
    Mzr = Dr * (cos(atan(Br*alpha__r))) * cos(alpha_vec(i));
    t = Dt * cos(Ct * atan(Bt * alpha__t - Et * (Bt * alpha__t - atan(Bt * alpha__t)))) * cos(alpha_vec(i));

    Mz0_vec(i) = -fy0 * t + Mzr;
end
end