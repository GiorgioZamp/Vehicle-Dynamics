% Combined longitudinal force FX
function [fx_vec] = MF96_FXcomb_vect(kappa_vec, alpha_vec, phi_vec, Fz_vec, tyre_data)

fx_vec = zeros(size(kappa_vec));
for i = 1:length(kappa_vec)

    [Gxa,~,~] = MF96_FXFYCOMB_coeffs(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    fx0 = MF96_FX0_vec(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    fx_vec(i) = Gxa * fx0;

end
end