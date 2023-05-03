function res = resid_pure_Mz_gamma(P,MZ,ALPHA,GAMMA,FZ,tyre_data)

    % ----------------------------------------------------------------------
    % Compute the residuals - least squares approach - to fit the Mz curve 
    %  with Fz=Fz_nom, IA=var. Pacejka 1996 Magic Formula
    % ----------------------------------------------------------------------

    % Define Magic Formula coefficients    
    tmp_tyre_data = tyre_data;
    
    tmp_tyre_data.qHz3  = P(1) ;
    tmp_tyre_data.qHz4  = P(2) ;
    tmp_tyre_data.qBz4  = P(3) ;
    tmp_tyre_data.qBz5  = P(4) ;
    tmp_tyre_data.qDz3  = P(5) ;
    tmp_tyre_data.qDz4  = P(6) ;
    tmp_tyre_data.qEz5  = P(7) ;
    tmp_tyre_data.qDz8  = P(8) ;
    tmp_tyre_data.qDz9  = P(9) ;

    res = 0;
    for i=1:length(ALPHA)
       Mz0  = MF96_Mz0(0, ALPHA(i), GAMMA(i), FZ(i), tmp_tyre_data);
       res = res+(Mz0-MZ(i))^2;
    end
    
    % Compute the residuals
    res = res/sum(MZ.^2);
    
end
