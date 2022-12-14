function plot_action_value_composite_task_agent

close all
clear all
clc

% Plot action value.

load('agent_behavior_composite_task.mat')

behavior_composite_task_pretraining = behavior_composite_task.pretraining;

% Initialize.
for theta_num = 1:8
    q_task1_function_agent{theta_num} = [];
    q_task2_function_agent{theta_num} = [];
    q_function_agent{theta_num} = [];
end
for lick_bin = 1:2
    q_task1_function_lick_agent{theta_num} = [];
    q_task2_function_lick_agent{theta_num} = [];
    q_function_lick_agent{lick_bin} = [];
end

for seed_num = 1:numel(behavior_composite_task_pretraining)
    clearvars -except behavior_composite_task_pretraining q_task1_function_agent q_task2_function_agent q_function_agent q_task1_function_lick_agent q_task2_function_lick_agent q_function_lick_agent seed_num
    
    q_input = behavior_composite_task_pretraining{seed_num}.late.q_input;
    q_task1_output = behavior_composite_task_pretraining{seed_num}.late.q_task1_output;
    q_task2_output = behavior_composite_task_pretraining{seed_num}.late.q_task2_output;
    
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
                q_task1_output_angle(x,y,theta_num) = nanmean(q_task1_output(find(binX_pos == x & binY_pos == y & bin_angle_combined' == theta_num)));
                q_task2_output_angle(x,y,theta_num) = nanmean(q_task2_output(find(binX_pos == x & binY_pos == y & bin_angle_combined' == theta_num)));
            end
        end
    end
    
    % Agent lick.
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_task1_output_lick(x,y,1) = nanmean(q_task1_output(find(binX_pos == x & binY_pos == y & q_input(:,7) <= 0.08)));
            q_task1_output_lick(x,y,2) = nanmean(q_task1_output(find(binX_pos == x & binY_pos == y & q_input(:,7) > 0.08)));
            q_task2_output_lick(x,y,1) = nanmean(q_task2_output(find(binX_pos == x & binY_pos == y & q_input(:,7) <= 0.08)));
            q_task2_output_lick(x,y,2) = nanmean(q_task2_output(find(binX_pos == x & binY_pos == y & q_input(:,7) > 0.08)));
        end
    end
    
    % Get action-value function.
    for theta_num = 1:8
        q_task1_function{theta_num} = imrotate(squeeze(q_task1_output_angle(:,:,theta_num)),90);
        q_task2_function{theta_num} = imrotate(squeeze(q_task2_output_angle(:,:,theta_num)),90);
    end
    for lick_bin = 1:2
        q_task1_function_lick{lick_bin} = imrotate(squeeze(q_task1_output_lick(:,:,lick_bin)),90);
        q_task2_function_lick{lick_bin} = imrotate(squeeze(q_task2_output_lick(:,:,lick_bin)),90);
    end
    
    % Concatenate across agents.
    for theta_num = 1:8
        q_function_agent{theta_num} = cat(3,q_function_agent{theta_num},(q_task1_function{theta_num} + q_task2_function{theta_num})./2);
    end
    for lick_bin = 1:2
        q_function_lick_agent{lick_bin} = cat(3,q_function_lick_agent{lick_bin},(q_task1_function_lick{lick_bin} + q_task2_function_lick{lick_bin})./2);
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
imagesc(filtered_mean_mean_q_function_agent_theta,[0.5,1])
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
    imagesc(mean_q_function_agent_mean_subtracted{theta_num},[-0.04,0.04])
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

% Mean across agents.
for lick_bin = 1:2
    mean_q_function_lick_agent{lick_bin} = nanmean(q_function_lick_agent{lick_bin},3);
end

% Mean across actions.
mean_q_function_lick_agent_lick = [];
for lick_bin = 1:2
    mean_q_function_lick_agent_lick = cat(3,mean_q_function_lick_agent_lick,mean_q_function_lick_agent{lick_bin});
end
mean_mean_q_function_lick_agent_lick = nanmean(mean_q_function_lick_agent_lick,3);

% Plot mean action values averaged across actions.
figure('Position',[500,500,250,250],'Color','w');
filtered_mean_mean_q_function_lick_agent_lick = nanconv(mean_mean_q_function_lick_agent_lick,image_filter,'edge','nanout');
imagesc(filtered_mean_mean_q_function_lick_agent_lick,[0.5,1])
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

% Plot average-subtracted mean action values.
for lick_bin = 1:2
    figure('Position',[500,500,250,250],'Color','w');
    mean_q_function_lick_agent_mean_subtracted{lick_bin} = nanconv(mean_q_function_lick_agent{lick_bin} - mean_mean_q_function_lick_agent_lick,image_filter,'edge','nanout');
    imagesc(mean_q_function_lick_agent_mean_subtracted{lick_bin},[0.002,0.008])
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
end

end