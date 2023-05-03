function res = resid_pure_Mz(P,MZ,alpha,gamma,fz,tyre_data)

    % ----------------------------------------------------------------------
    % Compute the residuals - least squares approach - to fit the Mz curve 
    %  with Fz=Fz_nom, IA=0. Pacejka 1996 Magic Formula
    % ----------------------------------------------------------------------

    % Define Magic Formula coefficients    
    tmp_tyre_data = tyre_data;
    
    tmp_tyre_data.qBz1  = P(1); 
    tmp_tyre_data.qBz9  = P(2);
    tmp_tyre_data.qBz10 = P(3);
    tmp_tyre_data.qCz1  = P(4);
    tmp_tyre_data.qDz1  = P(5);
    tmp_tyre_data.qDz6  = P(6);
    tmp_tyre_data.qEz1  = P(7);
    tmp_tyre_data.qEz4  = P(8);
    tmp_tyre_data.qHz1  = P(9);
    
    % Lateral Force Equations (Pure Side Slip)
    res = 0;
    for i=1:length(alpha)
       Mz0  = MF96_Mz0(0, alpha(i), gamma, fz, tmp_tyre_data);
       res = res+(Mz0-MZ(i))^2;
    end
    
    % Compute the residuals
    res = res/sum(MZ.^2);
    
end
