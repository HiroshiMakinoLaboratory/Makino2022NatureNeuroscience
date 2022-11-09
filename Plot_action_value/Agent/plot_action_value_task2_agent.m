function plot_action_value_task2_agent

close all
clear all
clc

% Plot action value.

load('agent_behavior_task2.mat')

% Initialize.
for lick_bin = 1:2
    q_function_lick_agent{lick_bin} = [];
end

for seed_num = 1:numel(behavior_task2)
    clearvars -except behavior_task2 q_function_lick_agent seed_num
    
    q_input = behavior_task2{seed_num}.late.q_input;
    q_output = behavior_task2{seed_num}.late.q_output;
    
    bin_size_pos = 40;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    % Agent lick.
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_output_lick_temp(x,y,1) = nanmean(q_output(find(binX_pos == x & binY_pos == y & q_input(:,7) <= 0.08)));
            q_output_lick_temp(x,y,2) = nanmean(q_output(find(binX_pos == x & binY_pos == y & q_input(:,7) > 0.08)));
        end
    end

    for lick_bin = 1:2
        q_output_lick_reward_zone{lick_bin} = q_output_lick_temp(17:24,1:4,lick_bin);
        q_output_lick_binned(lick_bin) = nanmean(q_output_lick_reward_zone{lick_bin}(:));
        q_output_lick{lick_bin} = nan(40,40);
        q_output_lick{lick_bin}(17:24,1:4) = q_output_lick_binned(lick_bin);
    end
    
    % Get action-value function.
    for lick_bin = 1:2
        q_function_lick{lick_bin} = imrotate(q_output_lick{lick_bin},90);
    end
    
    % Concatenate across agents.
    for lick_bin = 1:2
        q_function_lick_agent{lick_bin} = cat(3,q_function_lick_agent{lick_bin},q_function_lick{lick_bin});
    end
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

image_filter = fspecial('gaussian',2,2);

% Plot mean action values averaged across actions.
figure('Position',[500,500,250,250],'Color','w');
filtered_mean_mean_q_function_lick_agent_lick = nanconv(mean_mean_q_function_lick_agent_lick,image_filter,'edge','nanout');
imagesc(filtered_mean_mean_q_function_lick_agent_lick,[0.91,0.96])
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
    imagesc(mean_q_function_lick_agent_mean_subtracted{lick_bin},[-0.02,0.02])
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