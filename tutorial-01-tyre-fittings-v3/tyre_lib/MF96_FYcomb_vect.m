% Combined longitudinal force FX
function [fy_vec] = MF96_FYcomb_vect(kappa_vec, alpha_vec, phi_vec, Fz_vec, tyre_data)

fy_vec = zeros(size(alpha_vec));
for i = 1:length(kappa_vec)
    
    [~,Gyk,SVyk] = MF96_FXFYCOMB_coeffs(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    [fy0] = MF96_FY0(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    fy_vec(i) = Gyk * fy0 + SVyk;

end
end