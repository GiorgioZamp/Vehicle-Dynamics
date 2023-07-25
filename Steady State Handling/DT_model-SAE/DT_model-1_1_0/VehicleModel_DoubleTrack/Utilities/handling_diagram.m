%% Handling Diagram
% Function to compute all the characteristics requested in the assignment

function handling_diagram(model_sim,vehicle_data,flg)
    %% Auxiliary
    % ---------------------------------
    cc = jet(20); % color set
    %for i=1:20
    %   yline(i,'Color',cc(i,:))
    %   hold on
    %end

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
    %% Lateral Load Transfer
    % --------------------------
    e_phi = K_sf/(K_sf+K_sr); %Roll stiffness ratio
    fake_Ay = linspace(0,max(Ay_ss),length(Ay_ss))';

    dFz_f_n = m_s.*Ay_ss.*((Lr*h_rf)/(L*Wf) + h_s/Wf*e_phi); % Nominal
    dFz_r_n = m_s.*Ay_ss.*((Lf*h_rr)/(L*Wr) + h_s/Wr*(1-e_phi));
    dFz_f = 0.5.*abs(Fz_fr_ss-Fz_fl_ss); % Simulation
    dFz_r = 0.5.*abs(Fz_rr_ss-Fz_rl_ss);
    
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
    plot(Ay_ss,dFz_f_n)
    plot(Ay_ss,dFz_r_n)
    xlabel('$a_y [m/s^2]$')
    ylabel({'$\Delta Fz_f$,$\Delta Fz_r$ [N]'})
    legend('Front','Rear')
    title('Lateral Load Transfer in Ay')
    exportgraphics(f,'Graphs/LateralLoadTransfAy.eps')
    hold off
    % --------------------------
    %% Normalized Axle Characteristics
    % --------------------------------

    % Side Slip Angles
    alpha_f = 0.5.*deg2rad(alpha_fr_ss + alpha_fl_ss); % Simulated
    alpha_r = 0.5.*deg2rad(alpha_rr_ss + alpha_rl_ss); 

    alpha_f_n = delta_ss - beta_ss - rho_ss.*Lf; % Analytical
    alpha_r_n = -beta_ss + rho_ss.*Lr;
    Dalpha = alpha_r - alpha_f;

    % Axle Normal Loads
    Fz_f = Fz_fl_ss + Fz_fr_ss;
    Fz_r = Fz_rl_ss + Fz_rr_ss;

    % Simulation Axle Lateral Forces
    Fy_f = sin(delta_fl_ss).*Fx_fl_ss + cos(delta_fl_ss).*Fy_fl_ss + sin(delta_fr_ss).*Fx_fr_ss + cos(delta_fr_ss).*Fy_fr_ss;
    Fy_r = Fy_rr_ss + Fy_rl_ss;

    % Axle Characteristics
    Y_f = m*Ay_ss*Lr/L;
    Y_r = m*Ay_ss*Lf/L;

    % Normalized Axle Characteristics
    mu_f = Y_f./Fz_f;
    mu_r = Y_r./Fz_r;

    % Normalized Cornering Stiffnesses  -> they are close to zero
    Cy_f = diff(mu_f)./diff(alpha_f);
    Cy_r = diff(mu_r)./diff(alpha_r);

    % PLOTS
    % Axle Characteristics
    f = figure('Name','Axle Characteristics');
    tiledlayout(2,2)

    nexttile
    hold on
    plot(alpha_f,Fy_fr_ss,'Color',cc(1,:))
    plot(alpha_f,Fy_fl_ss,'Color',cc(6,:))
    xlabel('$\alpha_f [rad]$')
    ylabel({'$Fy_{fr}$,$Fy_{fl}$'})
    legend('$Fy_{fr}$','$Fy_{fl}$','Location','best')
    title('Lateral Forces Front')
    hold off

    nexttile
    hold on
    plot(alpha_r,Fy_rr_ss,'Color',cc(20,:))
    plot(alpha_r,Fy_rl_ss,'Color',cc(17,:))
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
    legend('$front$','$rear$','Location','northeast')
    hold off
    exportgraphics(f,'Graphs/NormAxleChar.eps')

    % --------------------------------
    %% Handling Diagram
    % --------------------
    % Steering Characteristics
    % ->  rho_ss*L-delta_f = alpha_r-alpha_f
    % ->  delta_Ack = rho_ss*L/tau_H   (when Dalpha=0)

    if flg == 1
        Dalpha_aux = -Dalpha;
        Dalpha_aux(1)=0;

        idx = normAy_ss<0.3;

        % Interpolate tangent
        p = polyfit(normAy_ss(idx),Dalpha_aux(idx),1);
        Kus_fit = p(1);
        linetg = polyval(p,normAy_ss);

        f = figure('Name','Steering Characteristics');
        tiledlayout(2,2)
        nexttile([1,2])
        hold on
        plot(normAy_ss,Dalpha_aux,'r','LineWidth',1.5)
        plot(normAy_ss,linetg,'b--')
        yline(0,'g','LineWidth',1)
        ylim('padded')
        xlabel('$\frac{a_y}{g}$')
        ylabel('$-\Delta\alpha$')
        %     legend('$\Delta\alpha$','$\rho_0 L-\delta$','Location','best')
        title('Handling Diagram')
        hold off


        % Curvature
        nexttile(3)
        hold on
        plot(normAy_ss,rho_ss)
        yline(deg2rad(mean(delta_D))./(L*tau_D),'g')
        hold off
        xlabel('$\frac{a_y}{g}$')
        ylabel('$\rho [1/m]$')
        xlim([0.02,inf])
        ylim('padded')
        title('Curvature')

        % Radius
        nexttile(4)
        plot(normAy_ss,1./rho_ss)
        yline(L/deg2rad(mean(delta_D))*tau_D,'g')
        xlabel('$\frac{a_y}{g}$')
        ylabel('$R [m]$')
        xlim([0.02,inf])
        ylim('padded')
        title('Radius')

        exportgraphics(f,'Graphs/SteeringChar.eps')

    elseif flg == 2
        Dalpha_aux = -Dalpha;
        Dalpha_aux(1)=0;

        idx = normAy_ss<0.35;

        % Interpolate tangent
        p = polyfit(normAy_ss(idx),Dalpha_aux(idx),1);
        Kus_fit = p(1);
        linetg = polyval(p,normAy_ss);

        f = figure('Name','Steering Characteristics');
        tiledlayout(2,2)
        nexttile([1,2])
        hold on
        plot(normAy_ss,Dalpha_aux,'r','LineWidth',1.5)
        plot(normAy_ss,linetg,'b--')
        yline(0,'g','LineWidth',1)
        ylim('padded')
        xlabel('$\frac{a_y}{g}$')
        ylabel('$-\Delta\alpha$')
        %     legend('$\Delta\alpha$','$\rho_0 L-\delta$','Location','best')
        title('Handling Diagram')
        hold off


        % Curvature
        nexttile(3)
        hold on
        plot(normAy_ss,rho_ss)
        yline(deg2rad(mean(delta_D))./(L*tau_D),'g')
        hold off
        xlabel('$\frac{a_y}{g}$')
        ylabel('$\rho [1/m]$')
        xlim([0.02,inf])
        ylim('padded')
        title('Curvature')

        % Radius
        nexttile(4)
        plot(normAy_ss,1./rho_ss)
        yline(L/deg2rad(mean(delta_D))*tau_D,'g')
        xlabel('$\frac{a_y}{g}$')
        ylabel('$R [m]$')
        xlim([0.02,inf])
        ylim('padded')
        title('Radius')


        exportgraphics(f,'Graphs/SteeringCharSteer.eps')

    else
        error('Test Flag Not Defined')
    end



    % --------------------
    %% Understeering Gradient
    % Compare theoretical and fitted Kus
    % --------------------------------

    % Theoretical
    Kus_prac = -diff(Dalpha)./diff(Ay_ss);
    Kus_prac_lin = Kus_prac(idx);
    Kus_prac_lin = Kus_prac_lin(end);
    Kus_th = (-1/(L*tau_H*g))*((1./Cy_r) - (1./Cy_f));
    Kus_th_lin = Kus_th(idx);
    Kus_th_lin = Kus_th_lin(end);

    disp(['Theoretical Understeering Gradient ','Kus_th = ',num2str(Kus_th_lin)])
    disp(['Computed Understeering Gradient','Kus_prac = ',num2str(Kus_prac_lin)])
    disp(['From Handling Diagram ','KUS = ',num2str(Kus_fit)])

    % figure
    % hold on 
    % plot(normAy_ss(1:end-1),Kus_prac)
    % plot(normAy_ss(1:end-1),Kus_th)
    % hold off

    % --------------------------------
    %% Yaw Rate Gain
    % slide99----------------------------
    YR_gain = Omega_ss./delta_ss;
    % YR_gain = u_ss./(L*(1+Kus_fit*u_ss.^2));
    % YR_gain = rho_ss.*u_ss./delta_ss;

    if flg == 1
        f = figure('Name','Yaw Rate Gain');
        plot(u_ss,YR_gain,'r')
        hold on
        plot(u_ss,u_ss./L,'g')
        xlabel('$u [m/s]$')
        ylabel('$\frac{\Omega}{\delta}$')
        legend('Vehicle','Neutral')
        title('Yaw Rate Gain')
        hold off
        exportgraphics(f,'Graphs/yawrategain.eps')
    elseif flg == 2
        f = figure('Name','Yaw Rate Gain');
        plot(u_ss,YR_gain,'r')
        hold on
        plot(u_ss,u_ss./L,'g')
        xlabel('$u [m/s]$')
        ylabel('$\frac{\Omega}{\delta}$')
        legend('Vehicle','Neutral')
        title('Yaw Rate Gain')
        hold off
        exportgraphics(f,'Graphs/YawRateGainSteer.eps')
    else
        error('Test Flag Not Defined')
    end

    % --------------------------------
    %% Body Slip Gain
    % slide101--------------------------
    BS_gain = beta_ss./delta_ss;
    BS_neutral = Lr/L - alpha_f./delta_ss;

    if flg == 1
        f = figure('Name','Body Slip Gain');
        plot(u_ss,BS_gain)
        hold on
        plot(u_ss,BS_neutral,'g')
        xlabel('$u [m/s]$')
        ylabel('$\frac{\beta}{\delta}$')
        legend('Vehicle','Neutral')
        title('Body Slip Gain')
        hold off
        exportgraphics(f,'Graphs/bodyslipgain.eps')
    elseif flg == 2
        f = figure('Name','Body Slip Gain');
        plot(u_ss,BS_gain)
        hold on
        plot(u_ss,BS_neutral,'g')
        xlabel('$u [m/s]$')
        ylabel('$\frac{\beta}{\delta}$')
        legend('Vehicle','Neutral')
        title('Body Slip Gain')
        hold off
        exportgraphics(f,'Graphs/BodySlipGainSteer.eps')
    else
        error('Test Flag Not Defined')
    end

    % --------------------------------
end