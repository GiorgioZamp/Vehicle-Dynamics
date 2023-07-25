function rollstiff_effect(e_phi,vehicle_data)
% Vary stiffness distribution from front to back
% e_phi = K_sf/(K_sf+K_sr); %Roll stiffness ratio

leg = cell(size(e_phi)); %initialize for speed
datasets = cell(size(e_phi));

% get total stiff.
K_sf = vehicle_data.front_suspension.Ks_f;
K_sr = vehicle_data.rear_suspension.Ks_r;
K_tot = K_sf+K_sr;

for i=1:length(e_phi)
    
    % Set parameters
    % Get new axle stiffnesses
    vehicle_data.front_suspension.Ks_f = e_phi(i)*(K_tot); % [N/m] Front suspension+tire stiffness
    vehicle_data.rear_suspension.Ks_r  = (1-e_phi(i))*(K_tot); % [N/m] Rear suspension+tire stiffness

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
cc = winter(length(e_phi));

% Handling Diagram
f = figure('Name','Roll Stiffness Effect');
hold on
for i = 1:length(e_phi)
    plot(datasets{1,i}.nAy_n, -datasets{1,i}.Dalpha,'Color',cc(i,:))
    leg{i} = ['$\epsilon_{\phi}\;$',num2str(e_phi(i))];
end
hold off
xlabel('$\frac{a_y}{g}$')
ylabel('$-\Delta\alpha$')
legend(leg)
title('Handling Diagram in $\epsilon_{\phi}$')
exportgraphics(f,'Graphs/RollStiffEffectSpeed.eps')

%------------------------------------------------------------------------

end