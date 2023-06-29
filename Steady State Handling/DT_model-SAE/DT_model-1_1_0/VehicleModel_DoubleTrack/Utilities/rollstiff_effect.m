function rollstiff_effect(e_phi,vehicle_data)
% Vary stiffness distribution from front to back
% e_phi = K_sf/(K_sf+K_sr); %Roll stiffness ratio

leg = cell(size(e_phi)); %initialize for speed
datasets = cell(size(e_phi));

for i=1:length(e_phi)
    
    % Set parameters
    % get total stiff.
    K_sf = vehicle_data.front_suspension.Ks_f;
    K_sr = vehicle_data.rear_suspension.Ks_r;
    % get new axle stiff.
    vehicle_data.front_suspension.Ks_f = e_phi(i)*(K_sf+K_sr); % [N/m] Front suspension+tire stiffness
    vehicle_data.rear_suspension.Ks_r  = (1-e_phi(i))*(K_sf+K_sr); % [N/m] Rear suspension+tire stiffness

    % Simulate
    model_sim = sim('Vehicle_Model_2Track_OLD');

    % Extract data
    [handling_data] = extract_handling(model_sim,vehicle_data);

    % Store the dataset in the structure
    datasets{i} = handling_data;

    % show progress
    fprintf('Simulation %d/%d Completed\n',i,length(e_phi))
    
    % handling_diagram(model_sim,vehicle_data);
end
%------------------------------------------------------------------------

% Plots
cc = jet(length(e_phi));
%------------------------------------------------------------------------

% Handling Diagram
f = figure('Name','Roll Stiffness Effect');
hold on
for i = 1:length(e_phi)
    plot(datasets{1,i}.Ay_n(20000:end)./9.81, -datasets{1,i}.Dalpha(20000:end),'Color',cc(i,:))
    leg{i} = ['$\epsilon_{\phi}$',num2str(e_phi(i))];
end
xlabel('$\frac{a_y}{g}$')
ylabel('$-\Delta\alpha$')
legend(leg)
title('Handling Diagram in $\epsilon_{\phi}$')
exportgraphics(f,'Graphs/RollStiffEffect.eps')
%------------------------------------------------------------------------

% Lateral Forces
figure('Name','Lateral Load Transfer');
subplot(1,2,1)
hold on
for i = 1:length(e_phi)
    plot(datasets{1,i}.Ay_n(20000:end), datasets{1,i}.dFz_f(20000:end),'b')
end
xlabel('$a_y$')
ylabel('$dFz_f$')
title('Lateral Load Transfer Front')
hold off

subplot(1,2,2)
hold on
for i = 1:length(e_phi)
    plot(datasets{1,i}.Ay_n(20000:end), datasets{1,i}.dFz_r(20000:end),'r')
end
xlabel('$a_y$')
ylabel('$dFz_r$')
title('Lateral Load Transfer Rear')
hold off
%------------------------------------------------------------------------

end