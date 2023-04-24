% Pure lateral force FY0
function [fy0_vec] = MF96_FY0_MOD(P,kappa_vec, alpha_vec, phi_vec, Fz_vec, tyre_data)

% precode

tmp_tyre_data = tyre_data;

tmp_tyre_data.pDy2 = P(1);
tmp_tyre_data.pEy2 = P(2);
tmp_tyre_data.pEy3 = P(3);
tmp_tyre_data.pHy2 = P(4);
tmp_tyre_data.pVy2 = P(5);

fy0_vec = zeros(size(alpha_vec));
for i = 1:length(alpha_vec)

    [alpha__y, By, Cy, Dy, Ey, ~, SVy] = MF96_FY0_coeffs(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tmp_tyre_data);

    % main code

    fy0_vec(i) = magic_formula(alpha__y, By, Cy, Dy, Ey, SVy);

end
end