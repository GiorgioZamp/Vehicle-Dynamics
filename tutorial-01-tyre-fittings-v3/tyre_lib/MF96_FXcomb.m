% Combined longitudinal force FX
function [fx] = MF96_FXcomb(kappa, alpha, phi, Fz, tyre_data)

 % precode

  [Gxa,~,~] = MF96_FXFYCOMB_coeffs(kappa, alpha, phi, Fz, tyre_data);
  [fx0] = MF96_FX0(kappa, alpha, phi, Fz, tyre_data);
 % main code

  fx = Gxa * fx0;
  
 end
