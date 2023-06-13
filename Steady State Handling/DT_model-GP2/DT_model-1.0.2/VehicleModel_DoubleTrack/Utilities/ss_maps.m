function ss_maps(model_sim,vehicle_data,Ts)
% Build steady-state maps
%% Import Data
%% Compute Parameters
ay = min(Ay):1e-3:max(Ay);
d0 = 0:1e-2:0.1; % delta_0 in radians
rho_0 =@(ay,d0) ;
rho = zeros(length(ay),length(d0));
for i = 1:length(d0)
    rho = rho_0(ay,d0(i));
end
%% Plots
figure('Name','$a_y - \delta_0$'), hold on;
plot(rho) % analitical values
plot() % ramp steer
end
% DO THE SAME USING AY STEPS AND LINERLY VARYING DELTA