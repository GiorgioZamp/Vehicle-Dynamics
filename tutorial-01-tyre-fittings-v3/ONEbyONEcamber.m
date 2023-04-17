%% Fit coefficient with VARIABLE CAMBER
% Zero longitudinal slip k and fixed normal load Fz

% Initialise values for parameters to be optimised
%    [pDy3,pEy3,pEy4,pHy3,pKy3,pVy3,pVy4]
P0 = [ 1,   1,   1,   1,   1,   1,   1  ];
lb = [];
ub = [];
tmp = cell(5,1);
SA_vec = cell(5,1);
for i= 1:5
    if i==1
        strct = intersect_table_data(GAMMA_0,FZ_220);
    elseif i==2
        strct = intersect_table_data(GAMMA_1,FZ_220);
    elseif i==3
        strct = intersect_table_data(GAMMA_2,FZ_220);
    elseif i==4
        strct = intersect_table_data(GAMMA_3,FZ_220);
    elseif i==5
        strct = intersect_table_data(GAMMA_4,FZ_220);
    end

    ALPHA_vec = strct.SA; % extract for clarity
    GAMMA_vec = strct.IA;
    FY_vec    = strct.FY;
    SA_vec{i} = min(ALPHA_vec):0.001:max(ALPHA_vec); % side slip vector [rad]

    % Optimize the coefficients
    [P_varGamma,~,~] = fmincon(@(P)resid_pure_Fy_varGamma(P,FY_vec,ALPHA_vec,GAMMA_vec,tyre_coeffs.FZ0,tyre_coeffs),...
        P0,[],[],[],[],lb,ub);

    % Change tyre data with new optimal values
    tyre_coeffs.pDy3 = P_varGamma(1);
    tyre_coeffs.pEy3 = P_varGamma(2);
    tyre_coeffs.pEy4 = P_varGamma(3);
    tyre_coeffs.pHy3 = P_varGamma(4);
    tyre_coeffs.pKy3 = P_varGamma(5);
    tyre_coeffs.pVy3 = P_varGamma(6);
    tyre_coeffs.pVy4 = P_varGamma(7);

    % Use Magic Formula to compute the fitting function
    zeros_vec = zeros(size(SA_vec{i}));
    ones_vec  = ones(size(SA_vec{i}));

    
    tmp{i} = MF96_FY0_vec(zeros_vec, SA_vec{i}, mean(GAMMA_vec).*ones_vec, tyre_coeffs.FZ0*ones_vec,tyre_coeffs);
end

% Plot Raw Data and Fitted Function
figure('Name','Fy0 vs Gamma')
plot(FZ_220.SA*to_deg,FZ_220.FY,'.')
hold on
tmp_gamma = [0,1,2,3,4];
leg = cell(length(tmp_gamma)+1,1);
leg{1} = 'Raw Data';
for i = 1:5
    plot(SA_vec{i}*to_deg,tmp{i},'-') %,
    leg{i+1} = ['Fitted $\gamma$= ',num2str(tmp_gamma(i)),' [°]'];
end
xlabel('$\alpha$ [deg]')
ylabel('$F_{y0}$ [N]')
legend(leg,Location="best")
hold off