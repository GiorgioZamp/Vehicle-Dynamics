%% Handling Diagram
% Function to compute all the characteristics requested in the assignment

function handling_diagram(model_sim,vehicle_data,Ts)
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
    dt = time_sim(2)-time_sim(1);

    % -----------------
    % Inputs
    % -----------------
    ped_0      = model_sim.inputs.ped_0.data;
    delta_D    = model_sim.inputs.delta_D.data;

    % -----------------
    % States
    % -----------------
    x_CoM      = model_sim.states.x.data;
    y_CoM      = model_sim.states.y.data;
    psi        = model_sim.states.psi.data;
    u          = model_sim.states.u.data;
    v          = model_sim.states.v.data;
    Omega      = model_sim.states.Omega.data;
    Fz_rr      = model_sim.states.Fz_rr.data;
    Fz_rl      = model_sim.states.Fz_rl.data;
    Fz_fr      = model_sim.states.Fz_fr.data;
    Fz_fl      = model_sim.states.Fz_fl.data;
    delta      = model_sim.states.delta.data;
    omega_rr   = model_sim.states.omega_rr.data;
    omega_rl   = model_sim.states.omega_rl.data;
    omega_fr   = model_sim.states.omega_fr.data;
    omega_fl   = model_sim.states.omega_fl.data;
    alpha_rr   = model_sim.states.alpha_rr.data;
    alpha_rl   = model_sim.states.alpha_rl.data;
    alpha_fr   = model_sim.states.alpha_fr.data;
    alpha_fl   = model_sim.states.alpha_fl.data;
    kappa_rr   = model_sim.states.kappa_rr.data;
    kappa_rl   = model_sim.states.kappa_rl.data;
    kappa_fr   = model_sim.states.kappa_fr.data;
    kappa_fl   = model_sim.states.kappa_fl.data;

    % -----------------
    % Extra Parameters
    % -----------------
    Tw_rr      = model_sim.extra_params.Tw_rr.data;
    Tw_rl      = model_sim.extra_params.Tw_rl.data;
    Tw_fr      = model_sim.extra_params.Tw_fr.data;
    Tw_fl      = model_sim.extra_params.Tw_fl.data;
    Fx_rr      = model_sim.extra_params.Fx_rr.data;
    Fx_rl      = model_sim.extra_params.Fx_rl.data;
    Fx_fr      = model_sim.extra_params.Fx_fr.data;
    Fx_fl      = model_sim.extra_params.Fx_fl.data;
    Fy_rr      = model_sim.extra_params.Fy_rr.data;
    Fy_rl      = model_sim.extra_params.Fy_rl.data;
    Fy_fr      = model_sim.extra_params.Fy_fr.data;
    Fy_fl      = model_sim.extra_params.Fy_fl.data;
    Mz_rr      = model_sim.extra_params.Mz_rr.data;
    Mz_rl      = model_sim.extra_params.Mz_rl.data;
    Mz_fr      = model_sim.extra_params.Mz_fr.data;
    Mz_fl      = model_sim.extra_params.Mz_fl.data;
    gamma_rr   = model_sim.extra_params.gamma_rr.data;
    gamma_rl   = model_sim.extra_params.gamma_rl.data;
    gamma_fr   = model_sim.extra_params.gamma_fr.data;
    gamma_fl   = model_sim.extra_params.gamma_fl.data;
    delta_fr   = model_sim.extra_params.delta_fr.data;
    delta_fl   = model_sim.extra_params.delta_fl.data;

    % Chassis side slip angle beta [rad]
    beta = atan(v./u);

    % -----------------
    % Accelerations
    % -----------------
    % Derivatives of u, v [m/s^2]
    dot_u = diff(u)/Ts;
    dot_v = diff(v)/Ts;
    % Total longitudinal and lateral accelerations
    Ax = dot_u(1:end) - Omega(2:end).*v(2:end);
    Ay = dot_v(1:end) + Omega(2:end).*u(2:end);
    % Ax low-pass filtered signal (zero-phase digital low-pass filtering)
    Wn_filter = 0.01;
    [b_butt,a_butt] = butter(4,Wn_filter,'low');
    Ax_filt = filtfilt(b_butt,a_butt,Ax);  
    dot_u_filt = filtfilt(b_butt,a_butt,dot_u);  
    % Steady state lateral acceleration
    Ay_ss = Omega.*u;
    % Longitudinal jerk [m/s^3]
    jerk_x = diff(dot_u)/Ts;

    % -----------------
    % Other parameters
    % -----------------
    % Total CoM speed [m/s]
    vG = sqrt(u.^2 + v.^2);
    % Steady state and transient curvature [m]
    rho_ss   = Omega./vG;
    rho_tran = ((dot_v.*u(1:end-1) - dot_u.*v(1:end-1)) ./ ((vG(1:end-1)).^3)) + rho_ss(1:end-1);
    % Desired sinusoidal steering angle for the equivalent single track front wheel
    desired_steer_atWheel = delta_D/tau_D;


    % ---------------------------------
    %% Lateral Load Transfer
    % --------------------------
    e_phi = K_sf/(K_sf+K_sr); %Roll stiffness ratio
    
    dFz_f_n = m_s.*Ay_ss.*((Lr*h_rf)/(L*Wf) + h_s/Wf*e_phi);
    dFz_r_n = m_s.*Ay_ss.*((Lf*h_rr)/(L*Wr) + h_s/Wr*(1-e_phi));
    dFz_f = 0.5.*abs(Fz_fr-Fz_fl);
    dFz_r = 0.5.*abs(Fz_rr-Fz_rl);
    
    % Plot
    f = figure('Name','Lateral Load Transfer in t');
    hold on
    plot(time_sim,dFz_f_n,'r--')
    plot(time_sim,dFz_r_n,'b--')
    plot(time_sim,dFz_f,'r')
    plot(time_sim,dFz_r,'b')
    xlabel('$t [s]$')
    ylabel({'$\Delta F_zf$,$\Delta F_zr$ [N]'})
    legend('FN','RN','FR','RR','Location','best')
    title('Lateral Load Transfer')
    exportgraphics(f,'Graphs/LateralLoadTransf.eps')
    hold off

%     f = figure('Name','Lateral Load Transfer in Ay');
%     hold on
%     plot(Ay_ss,dFz_f)
%     plot(Ay_ss,dFz_r)
%     xlabel('$a_y [m/s^2]$')
%     ylabel({'$\Delta F_zf$,$\Delta F_zr$ [N]'})
%     legend('Front','Rear')
%     title('Lateral Load Transfer in Ay')
% %     exportgraphics(f,'Graphs/LateralLoadTransf.eps')
%     hold off
    % --------------------------
    %% Normalized Axle Characteristics
    % --------------------------------
    % Side Slip Angles
    alpha_f = 0.5.*(alpha_fr+alpha_fl); % delta-beta-rho_ss.*Lf;
    alpha_r = 0.5.*(alpha_rr+alpha_rl); % -beta + rho_ss.*Lr;   %Not sure about this
    Dalpha = alpha_r-alpha_f;

    % Axle Normal Loads
    Fz_f = Fz_fl+Fz_fr;
    Fz_r = Fz_rl+Fz_rr;

    % Axle Characteristics
    Y_f = m*Ay_ss*Lr/L;
    Y_r = m*Ay_ss*Lf/L;

    % Real Axle Lateral Forces
    Fy_f = Fy_fr+Fy_fl;
    Fy_r = Fy_rr+Fy_rl;

    % Normalized Axle Characteristics
    mu_f = Y_f./Fz_f;
    mu_r = Y_r./Fz_r;

    % Plots FIX X AXIS -> NOT CORRECT REPRESENTATION
    f = figure('Name',' Axle Characteristics');
    tiledlayout(2,2)
    nexttile
    hold on
    plot(linspace(0,max(alpha_f),length(Y_f)),Y_f,'b')
    plot(linspace(0,max(alpha_f),length(Fy_f)),Fy_f,'-.','Color',[0 0.4470 0.7410])
    xlabel('$\alpha [rad]$')
    ylabel({'$Y_f$,$Fy_f$'})
    legend('Yf','$Fy_f$','Location','best')
    title('Axle Characteristics Front')
    hold off

    nexttile
    hold on
    plot(linspace(0,max(abs(alpha_r)),length(Y_r)),Y_r,'r')
    plot(linspace(0,max(abs(alpha_r)),length(Fy_r)),Fy_r,'-.','Color',[0.957 0.263 0.212])
    xlabel('$\alpha [rad]$')
    ylabel({'$Y_r$,$Fy_r$'})
    legend('Yr','$Fy_r$','Location','best')
    title('Axle Characteristics Rear')
    hold off

    nexttile([1,2])
    hold on
    plot(linspace(0,0.004,length(mu_f)),mu_f,'b')
    plot(linspace(0,0.004,length(mu_r)),mu_r,'r')
    xlabel('$\alpha [rad]$')
    ylabel({'$\mu_f$,$\mu_r$'})
    title('Normalized Axle Characteristics')
    exportgraphics(f,'Graphs/NormAxleChar.eps')

    % --------------------------------
    %% Handling Diagram
    % --------------------
    % Steering Characteristics
    %rho_ss*L = delta + (alpha_r-alpha_f)
    delta_f = deg2rad(mean(delta_D))*tau_D;
    rho = (delta_f + Dalpha)./L;

%     figure,hold on;
%     plot()
%     plot()
%     xlabel('')
%     ylabel('')
%     legend('','')
%     hold off
    
    % --------------------
    %% Understeering Gradient
    % Compare theoretical and fitted Kus
    % --------------------------------
    % Theoretical

    % Fitted

    % --------------------------------
    %% Yaw Rate Gain
    % slide99----------------------------
    YR_gain = rho_ss.*u./delta; %Omega./delta;

    figure('Name','Yaw Rate Gain')
    plot(u,YR_gain)
    hold on
    plot(u,Omega./u,'g')
    xlabel('$u [m/s]$')
    ylabel('$\frac{\Omega}{\delta}$')
    legend('V','N')
    title('Yaw Rate Gain')
    hold off

    % --------------------------------
    %% Body Slip Gain
    % slide101---------------------------
    BS_gain = beta./delta;
% 
%     figure('Name','Body Slip Gain')
%     plot(u,BS_gain)
%     hold on
%     plot(,'g')
%     xlabel('$u [m/s]$')
%     ylabel('$\frac{\Beta}{\delta}$')
%     legend('V','N')
%     title('Body Slip Gain')
%     hold off

    % --------------------------------
end