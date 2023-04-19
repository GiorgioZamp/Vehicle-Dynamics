function res = resid_comb_Fy(P,fy0,FY,KAPPA,ALPHA,FZ,tyre_data)

    % ----------------------------------------------------------------------
    %% Compute the residuals - least squares approach - to fit the Fy Combined curve 
    %  with Fz=Fz_nom, IA=0. Pacejka 1996 Magic Formula
    % ----------------------------------------------------------------------

    % Define MF coefficients

    %Fz0 = 200*4.44822; % Nominal load 200 lbf
    
    tmp_tyre_data = tyre_data;
    tmp_tyre_data.rBy1 = P(1); 
    tmp_tyre_data.rBy2 = P(2);  
    tmp_tyre_data.rBy3 = P(3); 
    tmp_tyre_data.rCy1 = P(4);
    tmp_tyre_data.rHy1 = P(5);
    tmp_tyre_data.rVy1 = P(6);
    tmp_tyre_data.rVy4 = P(7);
    tmp_tyre_data.rVy5 = P(8);
    tmp_tyre_data.rVy6 = P(9);

    % Lateral Force (Combined Slip) Equations
    res = 0;
    for i=1:length(ALPHA)
       fy  = MF96_FYcomb(fy0(i), KAPPA(i), ALPHA(i), 0, FZ, tmp_tyre_data);
       res = res+(fy-FY(i))^2;
    end
    
    % Compute the residuals
    res = res/sum(FY.^2);
    
end
