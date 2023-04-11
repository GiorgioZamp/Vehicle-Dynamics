% Self Aligning Moment Mz0
function [Mz0] = MF96_Mz0(fy, SHy, SVy, kappa, alpha, phi, Fz, tyre_data)

 % precode

  [alpha__t, Bt, Ct, Dt, Et, Br, Dr, alpha__r] = MF96_MZ0_coeffs(kappa, alpha, phi, Fz, tyre_data, SHy, SVy)

 % main code

  Mzr = Dr * ((Br ^ 2 * alpha__r ^ 2 + 1) ^ (-0.1e1 / 0.2e1)) * cos(alpha);
  
  t = Dt * cos(Ct * atan(-Bt * alpha__t + Et * (Bt * alpha__t - atan(Bt * alpha__t)))) * cos(alpha);
  
  Mz0 = -t * fy + Mzr;
  
 end
