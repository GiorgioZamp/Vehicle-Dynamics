function [handling_data] = extract_handling(model_sim,vehicle_data)

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
    dt = time_sim(2)-time_sim(1);

    % cutting index
    idx = time_sim>21;

    time_sim = time_sim(idx);

    % -----------------
    % Inputs
    % -----------------
    ped_0      = model_sim.inputs.ped_0.data;
    delta_D    = model_sim.inputs.delta_D.data;

    delta_D_ss = delta_D(idx);

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
    % omega_rr   = model_sim.states.omega_rr.data;
    % omega_rl   = model_sim.states.omega_rl.data;
    % omega_fr   = model_sim.states.omega_fr.data;
    % omega_fl   = model_sim.states.omega_fl.data;
    alpha_rr   = model_sim.states.alpha_rr.data;
    alpha_rl   = model_sim.states.alpha_rl.data;
    alpha_fr   = model_sim.states.alpha_fr.data;
    alpha_fl   = model_sim.states.alpha_fl.data;
    % kappa_rr   = model_sim.states.kappa_rr.data;
    % kappa_rl   = model_sim.states.kappa_rl.data;
    % kappa_fr   = model_sim.states.kappa_fr.data;
    % kappa_fl   = model_sim.states.kappa_fl.data;

    u_ss = u(idx);
    v_ss = v(idx);
    Omega_ss = Omega(idx);
    Fz_rr_ss = Fz_rr(idx);
    Fz_rl_ss = Fz_rl(idx);
    Fz_fr_ss = Fz_fr(idx);
    Fz_fl_ss = Fz_fl(idx);
    alpha_rr_ss = alpha_rr(idx);
    alpha_rl_ss = alpha_rl(idx);
    alpha_fr_ss = alpha_fr(idx);
    alpha_fl_ss = alpha_fl(idx);

    % -----------------
    % Extra Parameters
    % -----------------
    % Tw_rr      = model_sim.extra_params.Tw_rr.data;
    % Tw_rl      = model_sim.extra_params.Tw_rl.data;
    % Tw_fr      = model_sim.extra_params.Tw_fr.data;
    % Tw_fl      = model_sim.extra_params.Tw_fl.data;
    Fx_rr      = model_sim.extra_params.Fx_rr.data;
    Fx_rl      = model_sim.extra_params.Fx_rl.data;
    Fx_fr      = model_sim.extra_params.Fx_fr.data;
    Fx_fl      = model_sim.extra_params.Fx_fl.data;
    Fy_rr      = model_sim.extra_params.Fy_rr.data;
    Fy_rl      = model_sim.extra_params.Fy_rl.data;
    Fy_fr      = model_sim.extra_params.Fy_fr.data;
    Fy_fl      = model_sim.extra_params.Fy_fl.data;
    % Mz_rr      = model_sim.extra_params.Mz_rr.data;
    % Mz_rl      = model_sim.extra_params.Mz_rl.data;
    % Mz_fr      = model_sim.extra_params.Mz_fr.data;
    % Mz_fl      = model_sim.extra_params.Mz_fl.data;
    % gamma_rr   = model_sim.extra_params.gamma_rr.data;
    % gamma_rl   = model_sim.extra_params.gamma_rl.data;
    % gamma_fr   = model_sim.extra_params.gamma_fr.data;
    % gamma_fl   = model_sim.extra_params.gamma_fl.data;
    delta_fr   = model_sim.extra_params.delta_fr.data;
    delta_fl   = model_sim.extra_params.delta_fl.data;
    delta_ss  = deg2rad(delta_D_ss*tau_H);

    Fx_rr_ss = Fx_rr(idx);
    Fx_rl_ss = Fx_rl(idx);
    Fx_fr_ss = Fx_fr(idx);
    Fx_fl_ss = Fx_fl(idx);
    Fy_rr_ss = Fy_rr(idx);
    Fy_rl_ss = Fy_rl(idx);
    Fy_fr_ss = Fy_fr(idx);
    Fy_fl_ss = Fy_fl(idx);
    delta_fr_ss = delta_fr(idx);
    delta_fl_ss = delta_fl(idx);

    % Chassis side slip angle beta [rad]
    beta = atan(v./u);
    beta_ss = atan(v_ss./u_ss);

    % -----------------
    % Accelerations
    % -----------------
    % % Total longitudinal and lateral accelerations
    % Ax = dot_u(1:end) - Omega(2:end).*v(2:end);
    % Ay = dot_v(1:end) + Omega(2:end).*u(2:end);
    % Steady state lateral acceleration
    Ay_ss = Omega_ss.*u_ss;
    normAy_ss = Ay_ss./g;

    % -----------------
    % Other parameters
    % -----------------
    % Total CoM speed [m/s]
    vG = sqrt(u_ss.^2 + v_ss.^2);
    % Steady state and transient curvature [m]
    rho_ss   = Omega_ss./vG;

   
    
    % ---------------------------------
    %% Handling Diagram
    % --------------------

    % Lateral Load Transfer
    % dFz_f = 0.5.*abs(Fz_fr-Fz_fl);
    % dFz_r = 0.5.*abs(Fz_rr-Fz_rl);

    % Lateral Forces
    % Fy_f = Fy_fr + Fy_fl;
    % Fy_r = Fy_rr + Fy_rl;

    % Axle Characteristics
    % Y_f = m*Ay_ss*Lr/L;
    % Y_r = m*Ay_ss*Lf/L;

    % Side Slip Angles
    alpha_f = 0.5.*deg2rad(alpha_fr_ss + alpha_fl_ss);
    alpha_r = 0.5.*deg2rad(alpha_rr_ss + alpha_rl_ss); 
    Dalpha = alpha_r - alpha_f;

    handling_data.nAy_n = normAy_ss;
    handling_data.Dalpha = Dalpha;
    % handling_data.dFz_f = dFz_f;
    % handling_data.dFz_r = dFz_r;


end