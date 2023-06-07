function axle_character(model_sim,vehicle_data,Ts)

%% Load Vehicle Data
    
    Lf = vehicle_data.vehicle.Lf;  % [m] Distance between vehicle CoG and front wheels axle
    Lr = vehicle_data.vehicle.Lr;  % [m] Distance between vehicle CoG and front wheels axle
    L  = vehicle_data.vehicle.L;   % [m] Vehicle length
    Wf = vehicle_data.vehicle.Wf;  % [m] Width of front wheels axle 
    Wr = vehicle_data.vehicle.Wr;  % [m] Width of rear wheels axle                   
    m  = vehicle_data.vehicle.m;   % [kg] Vehicle Mass
    g  = vehicle_data.vehicle.g;   % [m/s^2] Gravitational acceleration
    tau_D = vehicle_data.steering_system.tau_D;  % [-] steering system ratio (pinion-rack)

%% Load Simulation Data

    time_sim = model_sim.states.u.time;
    dt = time_sim(2)-time_sim(1);

    % Inputs
    ped_0      = model_sim.inputs.ped_0.data;
     delta_D    = model_sim.inputs.delta_D.data; % input steering angle

    % States
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

    % Extra Parameters
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

    % Accelerations
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

    % Other parameters
    % Total CoM speed [m/s]
    vG = sqrt(u.^2 + v.^2);
    % Steady state and transient curvature [m]
    rho_ss   = Omega./vG;
    rho_tran = ((dot_v.*u(1:end-1) - dot_u.*v(1:end-1)) ./ ((vG(1:end-1)).^3)) + rho_ss(1:end-1);
    % Desired sinusoidal steering angle for the equivalent single track front wheel
%     desired_steer_atWheel = delta_D/tau_D;

%% Lateral Load Transfer
Fz_r = Fz_rr + Fz_rl;
Fz_f = Fz_fr + Fz_fl;
dF_zr = (Fz_rr - Fz_rl)/2;
dF_zf = (Fz_fr - Fz_fl)/2;

figure('Name','Lateral Load Transfer')
plot(time_sim,dF_zr)
hold on
plot(time_sim,dF_zf)
xlabel('t[s]')
ylabel(['$\Delta Fz_f$',',','$\Delta Fz_r$'])
title('Lateral Load Transfer')
legend('R','F')
hold off

%% Axle Characteristics
% Lateral Forces
Fy_f = Fy_fr + Fy_fl; % front axle
Fy_r = Fy_rr + Fy_rl; % rear axle

% Y_f = m*Ay_ss*Lr/L; 
% Y_r = m*Ay_ss*Lf/L; 
% Normalized Axle Characteristics
Y_f = Fy_f./Fz_f; % Front Axle Characteristic
Y_r = Fy_r./Fz_r; % Rear Axle Characteristic

% Check equality
% plot(Y_f)
% hold on
% plot(Ay_ss/g)

% Axle Side Slips
Dalpha = -delta+rho_ss*L; % steering characteristics
alpha_f = +delta-beta-rho_ss*Lf;
alpha_r = -beta+rho_ss*Lr;




%% Plots
figure('Name','Axle Characteristics'), hold on;
plot(alpha_f,Fy_f,'LineWidth',2)
plot(alpha_r,Fy_r,'LineWidth',2)
xlabel(['$\alpha_f$ ',' $\alpha_f$'])
ylabel(['$F_{yf}$ ',' $F_{yr}$'])
legend('F','R')

figure('Name','Normalized Axle Char.'), hold on;
plot(alpha_f,Y_f,'LineWidth',2)
plot(alpha_r,Y_r,'LineWidth',2)
xlabel(['$\alpha_f$ ',' $\alpha_f$'])
ylabel(['$\mu_f$ ',' $\mu_r$'])
legend('F','R')

% figure('Name','Handling Map')
% plot(Ay_ss/g,rho_ss*L-delta_D*tau_D,'LineWidth',2)

end