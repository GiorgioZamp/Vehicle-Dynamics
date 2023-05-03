% Combined lateral force FY
function [fy] = MF96_FYcomb(kappa, alpha, phi, Fz, tyre_data)

 % precode

  [~,Gyk,SVyk] = MF96_FXFYCOMB_coeffs(kappa, alpha, phi, Fz, tyre_data);
  [fy0] = MF96_FY0(kappa, alpha, phi, Fz, tyre_data);
 % main code

  fy = Gyk * fy0 + SVyk;
  
 end
