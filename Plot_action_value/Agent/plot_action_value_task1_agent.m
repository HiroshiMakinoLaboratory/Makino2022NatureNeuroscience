function plot_action_value_task1_agent

close all
clear all
clc

% Plot action value.

load('agent_behavior_task1.mat')

% Initialize.
for theta_num = 1:8
    q_function_agent{theta_num} = [];
end

for seed_num = 1:numel(behavior_task1)
    clearvars -except behavior_task1 q_function_agent seed_num
    
    q_input = behavior_task1{seed_num}.late.q_input;
    q_output = behavior_task1{seed_num}.late.q_output;
    
    bin_size_pos = 40;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    theta_temp = atan(abs(q_input(:,6))./abs(q_input(:,5)));
    for q_input_num = 1:size(q_input,1)
        if q_input(q_input_num,5) >= 0 && q_input(q_input_num,6) >= 0
            theta(q_input_num) = theta_temp(q_input_num);
        elseif q_input(q_input_num,5) < 0 && q_input(q_input_num,6) >= 0
            theta(q_input_num) = pi - theta_temp(q_input_num);
        elseif q_input(q_input_num,5) < 0 && q_input(q_input_num,6) < 0
            theta(q_input_num) = pi + theta_temp(q_input_num);
        elseif q_input(q_input_num,5) >= 0 && q_input(q_input_num,6) < 0
            theta(q_input_num) = 2*pi - theta_temp(q_input_num);
        end
    end
    
    % Bin angle.
    [~,~,bin_angle] = histcounts(theta,[0:pi/8:2*pi]);
    
    % Combine bins.
    bin_angle_combined = zeros(1,size(q_input,1));
    bin_angle_combined(bin_angle == 1) = 1;
    bin_angle_combined(bin_angle == 2) = 2;
    bin_angle_combined(bin_angle == 3) = 2;
    bin_angle_combined(bin_angle == 4) = 3;
    bin_angle_combined(bin_angle == 5) = 3;
    bin_angle_combined(bin_angle == 6) = 4;
    bin_angle_combined(bin_angle == 7) = 4;
    bin_angle_combined(bin_angle == 8) = 5;
    bin_angle_combined(bin_angle == 9) = 5;
    bin_angle_combined(bin_angle == 10) = 6;
    bin_angle_combined(bin_angle == 11) = 6;
    bin_angle_combined(bin_angle == 12) = 7;
    bin_angle_combined(bin_angle == 13) = 7;
    bin_angle_combined(bin_angle == 14) = 8;
    bin_angle_combined(bin_angle == 15) = 8;
    bin_angle_combined(bin_angle == 16) = 1;
    
    % Agent movement.
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            for theta_num = 1:8
                q_output_angle(x,y,theta_num) = nanmean(q_output(find(binX_pos == x & binY_pos == y & bin_angle_combined' == theta_num)));
            end
        end
    end
    
    % Get action-value function.
    for theta_num = 1:8
        q_function{theta_num} = imrotate(squeeze(q_output_angle(:,:,theta_num)),90);
        q_function{theta_num}(13:28,13:28) = nan;
    end
    
    % Concatenate across agents.
    for theta_num = 1:8
        q_function_agent{theta_num} = cat(3,q_function_agent{theta_num},q_function{theta_num});
    end
end

% Mean across agents.
for theta_num = 1:8
    mean_q_function_agent{theta_num} = nanmean(q_function_agent{theta_num},3);
end

% Mean across actions.
mean_q_function_agent_theta = [];
for theta_num = 1:8
    mean_q_function_agent_theta = cat(3,mean_q_function_agent_theta,mean_q_function_agent{theta_num});
end
mean_mean_q_function_agent_theta = nanmean(mean_q_function_agent_theta,3);

image_filter = fspecial('gaussian',2,2);

% Plot mean action values averaged across actions.
figure('Position',[500,500,250,250],'Color','w');
filtered_mean_mean_q_function_agent_theta = nanconv(mean_mean_q_function_agent_theta,image_filter,'edge','nanout');
filtered_mean_mean_q_function_agent_theta(13:28,13:28) = 0.99;
imagesc(filtered_mean_mean_q_function_agent_theta,[0.75,1])
rectangle('Position',[12.5,12.5,16,16],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor','none')
xlabel('x (cm)');
ylabel('y (cm)')
xlim([0.5,40.5]);
ylim([0.5,40.5]);
axis square
ax = gca;
ax.Color = 'w';
ax.FontSize = 14;
ax.LineWidth = 1;
ax.XColor = 'k';
ax.YColor = 'k';
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;
ax.XTick = [0.5,20.5,40.5];
ax.XTickLabel = {'0','5','10'};
ax.YTick = [0.5,20.5,40.5];
ax.YTickLabel = {'10','5','0'};
colormap('redblue')

% Plot average-subtracted mean action values.
for theta_num = 1:8
    figure('Position',[500,500,250,250],'Color','w');
    mean_q_function_agent_mean_subtracted{theta_num} = nanconv(mean_q_function_agent{theta_num} - mean_mean_q_function_agent_theta,image_filter,'edge','nanout');
    imagesc(mean_q_function_agent_mean_subtracted{theta_num},[-0.03,0.03])
    rectangle('Position',[12.5,12.5,16,16],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor','none')
    xlabel('x (cm)');
    ylabel('y (cm)');
    xlim([0.5,40.5]);
    ylim([0.5,40.5]);
    axis square
    ax = gca;
    ax.Color = 'w';
    ax.FontSize = 14;
    ax.LineWidth = 1;
    ax.XColor = 'k';
    ax.YColor = 'k';
    ax.XLabel.FontSize = 14;
    ax.YLabel.FontSize = 14;
    ax.XTick = [0.5,20.5,40.5];
    ax.XTickLabel = {'0','5','10'};
    ax.YTick = [0.5,20.5,40.5];
    ax.YTickLabel = {'10','5','0'};
    colormap('redblue')
end

end