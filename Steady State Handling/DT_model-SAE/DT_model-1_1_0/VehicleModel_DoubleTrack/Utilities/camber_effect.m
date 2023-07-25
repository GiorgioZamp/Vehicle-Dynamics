function camber_effect(camber_set,vehicle_data)
% Camber Effect

% For camber test no toe used
vehicle_data.steering_system.delta__0 = 0;

leg = cell(size(camber_set)); %initialize for speed
datasets = cell(size(camber_set));

for i=1:length(camber_set)

    % Set parameters
    vehicle_data.front_wheel.static_camber = camber_set(i);
    vehicle_data.rear_wheel.static_camber = camber_set(i);

    % Simulate
    model_sim = sim('Vehicle_Model_2Track_OLD');

    % Extract data
    [handling_data] = extract_handling(model_sim,vehicle_data);

    % Store the dataset in the structure
    datasets{i} = handling_data;

    % Show progress
    fprintf('Simulation %d/%d Completed\n',i,length(camber_set))

end

% Plot
cc = jet(length(camber_set));
f = figure('Name','Camber Effect');
hold on;
for i = 1:length(camber_set)
    plot(datasets{1,i}.nAy_n, -datasets{1,i}.Dalpha,'Color',cc(i,:))
    leg{i} = ['$\gamma\;$',num2str(camber_set(i))];
end
xlabel('$\frac{a_y}{g}$')
ylabel('$-\Delta\alpha$')
legend(leg)
title('Handling Diagram in $\gamma$')
exportgraphics(f,'Graphs/CamberEffectSteer.eps');
hold off

end