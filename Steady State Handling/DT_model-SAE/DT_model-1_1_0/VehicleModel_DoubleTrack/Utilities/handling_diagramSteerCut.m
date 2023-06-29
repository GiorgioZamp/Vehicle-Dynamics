%% Handling Diagram
% Function to compute all the characteristics requested in the assignment

function handling_diagramSteerCut(model_sim,vehicle_data,Ts,cut_time)
    %% Auxiliary
    % ---------------------------------
    cc = jet(20); % color set

    % ---------------------------------
    %% Load vehicle data
    % ---------------------------------
    Lf = vehicle_data.vehicle.Lf;  % [m] Distance between vehicle CoG and front wheels axle
    Lr = vehicle_data.vehicle.Lr;  % [m] Distance between vehicle CoG and front wheels axle
    L  = vehicle_data.vehicle.L;   % [m] Vehicle length
    Wf = vehicle_data.vehicle.Wf;  % [m] Width of front wheels axle 
    Wr = vehicle_data.vehicle.Wr;  % [m] Width of rear wheels axle                   
    m  = vehicle_data.vehicle.m;   % [kg] Vehicle Mass
    m_s = 225;                     % [kg] Sprung Mass (Not present in struct)
    g  = vehicle_data.vehicle.g;   % [m/s^2] Gravitational acceleration
    tau_D = vehicle_data.steering_system.tau_D;  % [-] steering system ratio (pinion-rack)
    tau_H = 1/tau_D;
    h_G = vehicle_data.vehicle.hGs; % [m] CoM Height
    h_rf = vehicle_data.front_suspension.h_rc_f; % [m] front suspension roll height
    h_rr = vehicle_data.rear_suspension.h_rc_r;  % [m] rear suspension roll height
    h_r = h_rr+(h_rf-h_rr)*Lr/L; % [m] CoM height wrt RC
    h_s = h_G - h_r;
    K_sf = vehicle_data.front_suspension.Ks_f; % [N/m] Front suspension+tire stiffness
    K_sr = vehicle_data.rear_suspension.Ks_r; % [N/m] Rear suspension+tire stiffness

    % ---------------------------------
    %% Extract data from simulink model
    % ---------------------------------
    time_sim = model_sim.states.u.time;

    index = time_sim>cut_time; %cutting from 20 to end
    % ---------------------------------
    time_sim = time_sim(index);
    dt = time_sim(2)-time_sim(1);
    % -----------------
    % Inputs
    % -----------------
    ped_0      = model_sim.inputs.ped_0.data(index);
    delta_D    = model_sim.inputs.delta_D.data(index);

    % -----------------
    % States
    % -----------------
    x_CoM      = model_sim.states.x.data(index);
    y_CoM      = model_sim.states.y.data(index);
    psi        = model_sim.states.psi.data(index);
    u          = model_sim.states.u.data(index);
    v          = model_sim.states.v.data(index);
    Omega      = model_sim.states.Omega.data(index);
    Fz_rr      = model_sim.states.Fz_rr.data(index);
    Fz_rl      = model_sim.states.Fz_rl.data(index);
    Fz_fr      =    model_sim.states.Fz_fr.data(index);
    Fz_fl      =    model_sim.states.Fz_fl.data(index);
    delta      =    model_sim.states.delta.data(index);
    omega_rr   = model_sim.states.omega_rr.data(index);
    omega_rl   = model_sim.states.omega_rl.data(index);
    omega_fr   = model_sim.states.omega_fr.data(index);
    omega_fl   = model_sim.states.omega_fl.data(index);
    alpha_rr   = model_sim.states.alpha_rr.data(index);
    alpha_rl   = model_sim.states.alpha_rl.data(index);
    alpha_fr   = model_sim.states.alpha_fr.data(index);
    alpha_fl   = model_sim.states.alpha_fl.data(index);
    kappa_rr   = model_sim.states.kappa_rr.data(index);
    kappa_rl   = model_sim.states.kappa_rl.data(index);
    kappa_fr   = model_sim.states.kappa_fr.data(index);
    kappa_fl   = model_sim.states.kappa_fl.data(index);

    % -----------------
    % Extra Parameters
    % -----------------
    Tw_rr      = model_sim.extra_params.Tw_rr.data(index);
    Tw_rl      = model_sim.extra_params.Tw_rl.data(index);
    Tw_fr      = model_sim.extra_params.Tw_fr.data(index);
    Tw_fl      = model_sim.extra_params.Tw_fl.data(index);
    Fx_rr      = model_sim.extra_params.Fx_rr.data(index);
    Fx_rl      = model_sim.extra_params.Fx_rl.data(index);
    Fx_fr      = model_sim.extra_params.Fx_fr.data(index);
    Fx_fl      = model_sim.extra_params.Fx_fl.data(index);
    Fy_rr      = model_sim.extra_params.Fy_rr.data(index);
    Fy_rl      = model_sim.extra_params.Fy_rl.data(index);
    Fy_fr      = model_sim.extra_params.Fy_fr.data(index);
    Fy_fl      = model_sim.extra_params.Fy_fl.data(index);
    Mz_rr      = model_sim.extra_params.Mz_rr.data(index);
    Mz_rl      = model_sim.extra_params.Mz_rl.data(index);
    Mz_fr      = model_sim.extra_params.Mz_fr.data(index);
    Mz_fl      = model_sim.extra_params.Mz_fl.data(index);
    gamma_rr   = model_sim.extra_params.gamma_rr.data(index);
    gamma_rl   = model_sim.extra_params.gamma_rl.data(index);
    gamma_fr   = model_sim.extra_params.gamma_fr.data(index);
    gamma_fl   = model_sim.extra_params.gamma_fl.data(index);
    delta_fr   = model_sim.extra_params.delta_fr.data(index);
    delta_fl   = model_sim.extra_params.delta_fl.data(index);
    delta_use  = deg2rad(delta_fr+delta_fl)*0.5; % Average Steering Angle in radians

    % Chassis side slip angle beta [rad]
    beta = atan(v./u);

    % -----------------
    % Accelerations
    % -----------------
    % % Total longitudinal and lateral accelerations
    % Ax = dot_u(1:end) - Omega(2:end).*v(2:end);
    % Ay = dot_v(1:end) + Omega(2:end).*u(2:end);
    % Steady state lateral acceleration
    Ay_ss = Omega.*u;

    % -----------------
    % Other parameters
    % -----------------
    % Total CoM speed [m/s]
    vG = sqrt(u.^2 + v.^2);
    % Steady state and transient curvature [m]
    rho_ss   = Omega./vG;

    % ---------------------------------
    %% Cut in time
    index = time_sim>cut_time; %cutting from 20 to end

    % ---------------------------------
    %% Lateral Load Transfer
    % --------------------------
    e_phi = K_sf/(K_sf+K_sr); %Roll stiffness ratio
    fake_Ay = linspace(min(Ay_ss),max(Ay_ss),length(Ay_ss))';

    dFz_f_n = m_s.*fake_Ay.*((Lr*h_rf)/(L*Wf) + h_s/Wf*e_phi); % Nominal
    dFz_r_n = m_s.*fake_Ay.*((Lf*h_rr)/(L*Wr) + h_s/Wr*(1-e_phi));
    dFz_f = 0.5.*abs(Fz_fr-Fz_fl); % Simulation
    dFz_r = 0.5.*abs(Fz_rr-Fz_rl);
    
    % Plot
    f = figure('Name','Lateral Load Transfer in t');
    hold on
    plot(time_sim,dFz_f_n,'r--')
    plot(time_sim,dFz_r_n,'b--')
    plot(time_sim,dFz_f,'r')
    plot(time_sim,dFz_r,'b')
    xlabel('$t [s]$')
    ylabel({'$\Delta Fz_f$,$\Delta Fz_r$ [N]'})
    legend('FN','RN','FR','RR','Location','best')
    title('Lateral Load Transfer')
    exportgraphics(f,'Graphs/LateralLoadTransfT.eps')
    hold off

    f = figure('Name','Lateral Load Transfer in Ay');
    hold on
    plot(fake_Ay,dFz_f_n)
    plot(fake_Ay,dFz_r_n)
    xlabel('$a_y [m/s^2]$')
    ylabel({'$\Delta Fz_f$,$\Delta Fz_r$ [N]'})
    legend('Front','Ru [km/h]ear')
    title('Lateral Load Transfer in Ay')
    exportgraphics(f,'Graphs/LateralLoadTransfAy.eps')
    hold off
    % --------------------------
    %% Normalized Axle Characteristics
    % --------------------------------
    % Side Slip Angles
    alpha_f = 0.5.*deg2rad(alpha_fr + alpha_fl); % Simulated
    alpha_r = 0.5.*deg2rad(alpha_rr + alpha_rl); 

    alpha_f_n = delta_use - beta - rho_ss.*Lf; % Analytical
    alpha_r_n = -beta + rho_ss.*Lr;
    Dalpha = alpha_r - alpha_f;

    % Axle Normal Loads
    Fz_f = Fz_fl + Fz_fr;
    Fz_r = Fz_rl + Fz_rr;
%     Fz_f = m*g*Lr/L;
%     Fz_r = m*g*Lf/L;

    % Simulation Axle Lateral Forces
    Fy_f = Fy_fr + Fy_fl;
    Fy_r = Fy_rr + Fy_rl;

    % Computed Axle Lateral Forces
%     Fy_f_t = m.*fake_Ay.*Lr/L;
%     Fy_r_t = m.*fake_Ay.*Lf/L;

    % Axle Characteristics
    Y_f = m*Ay_ss*Lr/L;
    Y_r = m*Ay_ss*Lf/L;

    % Normalized Axle Characteristics
    mu_f = Y_f./Fz_f;
    mu_r = Y_r./Fz_r;

    % Normalized Cornering Stiffnesses
    Cy_f = gradient(mu_f);
    Cy_r = gradient(mu_r);

    % PLOTS
    % Axle Characteristics
    f = figure('Name','Axle Characteristics');
    tiledlayout(2,2)

    nexttile
    hold on
    plot(alpha_f,Fy_fr,'Color',cc(1,:))
    plot(alpha_f,Fy_fl,'Color',cc(6,:))
    xlabel('$\alpha_f [rad]$')
    ylabel({'$Fy_{fr}$,$Fy_{fl}$'})
    legend('$Fy_{fr}$','$Fy_{fl}$','Location','best')
    title('Lateral Forces Front')
    hold off

    nexttile
    hold on
    plot(alpha_r,Fy_rr,'Color',cc(20,:))
    plot(alpha_r,Fy_rl,'Color',cc(17,:))
    xlabel('$\alpha_r [rad]$')
    ylabel({'$Fy_{rr}$,$Fy_{rl}$'})
    legend('$Fy_{rr}$','$Fy_{rl}$','Location','best')
    title('Lateral Forces Rear')
    hold off

    nexttile([1,2])
    hold on
    plot(alpha_f,Fy_f,'b')
    plot(alpha_r,Fy_r,'r')
    xlabel('$\alpha [rad]$')
    ylabel({'$Fy_f$,$Fy_r$'})
    legend('$Fy_f$','$Fy_r$','Location','best')
    title('Axle Characteristics')
    exportgraphics(f,'Graphs/AxleChar.eps')

    % Theoretical Axle Characteristics
    f = figure('Name','Normalized Axle Characteristics');
    hold on
    plot(alpha_f,mu_f,'b')
    plot(alpha_r,mu_r,'r')
    xlabel('$\alpha_f , \alpha_r [rad]$')
    ylabel({'$\mu_f$,$\mu_r$'})
    title('Normalized Axle Characteristics')
    legend('$front$','$rear$','Location','best')
    hold off
    exportgraphics(f,'Graphs/NormAxleChar.eps')

    % --------------------------------
    %% Handling Diagram
    % --------------------
    % Steering Characteristics
    % ->  rho_ss*L-delta_f = alpha_r-alpha_f
    % ->  delta_Ack = rho_ss*L/tau_H   (when Dalpha=0)

    idx = Dalpha>6.2*1e-4;
    Ay_ss_aux = Ay_ss(idx); % cut
    Dalpha_aux = -Dalpha(idx); % cut

    % Interpolate tangent
    x_aux = [0,Ay_ss_aux(1)];
    y_aux = [0,Dalpha_aux(1)];
    p = polyfit(x_aux,y_aux,1);
    linetg = polyval(p,fake_Ay);

    f = figure('Name','Steering Characteristics');
    tiledlayout(2,2)

    nexttile([1,2])
    hold on
    plot(Ay_ss_aux./g,Dalpha_aux,'r','LineWidth',1.5)
    plot(fake_Ay./g,linetg,'b--')
    yline(0,'g','LineWidth',1)
    xlim([0,0.9])
    ylim('padded')
    xlabel('$\frac{a_y}{g}$')
    ylabel('$-\Delta\alpha$')
%     legend('$\Delta\alpha$','$\rho_0 L-\delta$','Location','best')
    title('Handling Diagram')
    hold off
    

    % Curvature
    nexttile(3)
    hold on
    plot(Ay_ss_aux/g,rho_ss(idx))
%     temp = gradient(rho_ss(idx));
%     tt = rho_ss(idx);
%     plot(Ay_ss_aux/g,temp(1).*Ay_ss_aux/g+tt,'b--')
    yline(deg2rad(mean(delta_D))/(L*tau_D),'g')
    hold off
    xlabel('$\frac{a_y}{g}$')
    ylabel('$\rho [1/m]$')
    ylim('padded')
    title('Curvature')
 
    % Radius
    nexttile(4)
    plot(Ay_ss_aux/g,1./rho_ss(idx))
    yline(L/deg2rad(mean(delta_D))*tau_D,'g')
    xlabel('$\frac{a_y}{g}$')
    ylabel('$R [m]$')
    ylim('padded')
    title('Radius')

    % Steering
%     nexttile
%     plot(Ay_ss_aux/g,delta_use(idx))
%     hold on
%     plot(Ay_ss_aux/g,rho_ss(idx)*L/tau_D,'g')
%     xlabel('$\frac{a_y}{g} [ ]$')
%     ylabel('$\delta [rad]$')
%     ylim('padded')
%     title('Steering')
%     hold off
    
    exportgraphics(f,'Graphs/SteeringChar.eps')
    % --------------------
    %% Understeering Gradient
    % Compare theoretical and fitted Kus
    % --------------------------------

    % Theoretical
    Kus = -m/(L^2)*(Lf/K_sr - Lr/K_sf);
    % -m/(L*tau_H)*(Lf/K_sr - Lr/K_sf);
    % -1/(L*tau_H*g)*(1/Cy_r - 1/Cy_f);

    disp('$K_{us} = $',num2str(Kus))
    disp('$K_{US} = $',num2str(p(1)))

    % Fitted
    % Kus_fit = gradient(delta_use(idx));
    % 
    % % Plot
    % f = figure('Name','Understeering Gradient');
    % hold on
    % plot(fake_Ay,Kus.*fake_Ay)
    % plot(fake_Ay,mean(Kus_fit(1:10)).*fake_Ay)
    % hold off
    % exportgraphics(f,'Graphs/UndersteeringGrad.eps')

    % p(1) is the slope of the tangent we computed in the handling
    % characteristics



    % --------------------------------
    %% Yaw Rate Gain
    % slide99----------------------------
    YR_gain = rho_ss.*u./delta_use; %Omega./delta;

    f = figure('Name','Yaw Rate Gain');
    plot(u(idx),YR_gain(idx),'r')
    hold on
    plot(u(idx),u(idx)./L,'g')
    xlabel('$u [m/s]$')
    ylabel('$\frac{\Omega}{\delta}$')
    legend('Vehicle','Neutral')
    title('Yaw Rate Gain')
    hold off
    exportgraphics(f,'Graphs/yawrategain.eps')

    % --------------------------------
    %% Body Slip Gain
    % slide101---------------------------
    BS_gain = beta./delta_use;
    beta_neutral = rad2deg(Lr/L*delta_use*tau_H - (alpha_f+alpha_r));

    f = figure('Name','Body Slip Gain');
    plot(u(idx),BS_gain(idx))
    hold on
    plot(u(idx),beta_neutral(idx),'g')
    xlabel('$u [m/s]$')
    ylabel('$\frac{\beta}{\delta}$')
    legend('Vehicle','Neutral')
    title('Body Slip Gain')
    hold off
    exportgraphics(f,'Graphs/bodyslipgain.eps')

    % --------------------------------
end