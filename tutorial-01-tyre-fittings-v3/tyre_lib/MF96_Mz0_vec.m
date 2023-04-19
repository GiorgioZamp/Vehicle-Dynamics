% Self Aligning Moment Mz0
% this function remap the scalar function to its vectorial form
function [Mz0_vec] = MF96_Mz0_vec(alpha_vec, Fz0, fy, SHy, SVy, tyre_data)


Mz0_vec = zeros(size(alpha_vec));
for i = 1:length(alpha_vec)
    % precode
    [alpha__t, Bt, Ct, Dt, Et, Br, Dr, alpha__r] = MF96_MZ0_coeffs(0, alpha_vec(i), 0, Fz0, SHy(i), SVy(i), tyre_data);
    % main code
    Mzr = Dr * ((Br ^ 2 * alpha__r ^ 2 + 1) ^ (-0.1e1 / 0.2e1)) * cos(alpha_vec(i));
    t = Dt * cos(Ct * atan(-Bt * alpha__t + Et * (Bt * alpha__t - atan(Bt * alpha__t)))) * cos(alpha_vec(i));

    Mz0_vec(i) = -fy(i) * t + Mzr;
end
end