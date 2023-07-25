function toe_effect(toe_set,vehicle_data)

    leg = cell(size(toe_set)); %initialize for speed
    datasets = cell(size(toe_set));

    for i=1:length(toe_set)

        % Set parameters (used inside vehicle model)
        delta__0 = deg2rad(toe_set(i)); % Use uniform toe angle
        vehicle_data.steering_system.delta__0 = delta__0;

        % Simulate
        model_sim = sim('Vehicle_Model_2Track_OLD');

        % Extract data
        [handling_data] = extract_handling(model_sim,vehicle_data);

        % Store the dataset in the structure
        datasets{i} = handling_data;
        
        % show progress
        fprintf('Simulation %d/%d Completed\n',i,length(toe_set))

    end

    % Plot
    cc = jet(length(toe_set));

    f = figure('Name','Toe Effect');
    hold on
    for i = 1:length(toe_set)
        plot(datasets{1,i}.nAy_n, -datasets{1,i}.Dalpha,'Color',cc(i,:))
        leg{i} = ['$\delta_0$ ',num2str(toe_set(i))];
    end
    hold off
    xlabel('$\frac{a_y}{g}$')
    ylabel('$-\Delta\alpha$')
    legend(leg)
    title('Handling Diagram in $\delta_0$')
    exportgraphics(f,'Graphs/ToeEffectSpeed.eps')

end