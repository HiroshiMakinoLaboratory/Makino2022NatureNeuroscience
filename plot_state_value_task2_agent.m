function plot_state_value_task2_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Plot state value.
% Input - Learning_stage: 'naive' or 'expert'.

load('agent_behavior_task2.mat')

% Initialize.
value_function_agent = [];
downsamp_value_function_agent = [];

for seed_num = 1:numel(behavior_task2)
    clearvars -except learning_stage behavior_task2 value_function_agent downsamp_value_function_agent seed_num
    
    if contains(learning_stage,'naive')
        observation = behavior_task2{seed_num}.naive.observation;
        action = behavior_task2{seed_num}.naive.action;
        reward = behavior_task2{seed_num}.naive.reward;
    elseif contains(learning_stage,'expert')
        observation = behavior_task2{seed_num}.expert.observation;
        action = behavior_task2{seed_num}.expert.action;
        reward = behavior_task2{seed_num}.expert.reward;
    end
    
    % Determine correct/incorrect trials.
    all_trial = [1:length(reward)];
    for trial_num = 1:length(reward)
        correct_trial_temp(trial_num) = reward{trial_num}(end);
    end
    correct_trial = find(correct_trial_temp);
    incorrect_trial = all_trial(~ismember(all_trial,correct_trial));
    
    % Calculate state values.
    for trial_num = 1:length(observation)
        x_trial{trial_num} = observation{trial_num}(:,1);
        y_trial{trial_num} = observation{trial_num}(:,2);
        
        % Get binned position.
        [~,~,~,x_bin{trial_num},y_bin{trial_num}] = histcounts2(x_trial{trial_num},y_trial{trial_num},'XBinEdges',[-1:0.1:1],'YBinEdges',[-1:0.1:1]);
        x_bin{trial_num} = x_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
        y_bin{trial_num} = y_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
    end
    
    % Get state-value function.
    gamma = 0.95; % Discount factor.
    for trial_num = 1:length(observation)
        for x_bin_num = 1:20
            for y_bin_num = 1:20
                mean_step_size_from_state(trial_num,x_bin_num,y_bin_num) = mean(gamma.^(length(x_bin{trial_num}) - find(x_bin{trial_num} == x_bin_num & y_bin{trial_num} == y_bin_num)));
            end
        end
    end
    
    % Incorporate miss trials.
    if ~isempty(incorrect_trial) == 1
        for incorrect_trial_num = 1:length(incorrect_trial)
            mean_step_size_from_state(incorrect_trial(incorrect_trial_num),:,:) = zeros(1,20,20);
        end
    end
    
    % Plot state value functions.
    value_function_temp = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
    image_filter = fspecial('gaussian',1,1);
    value_function = nanconv(value_function_temp,image_filter,'edge','nanout');
    
    % Concatenate across agents.
    value_function_agent = [value_function_agent,value_function(:)];
    
    % Downsample.
    value_function_temp1 = cat(3,value_function(1:2:end,1:2:end),value_function(2:2:end,2:2:end));
    value_function_temp2 = cat(3,value_function(1:2:end,2:2:end),value_function(2:2:end,1:2:end));
    downsamp_value_function = nanmean(cat(3,value_function_temp1,value_function_temp2),3);
    
    % Concatenate across agents.
    downsamp_value_function_agent = cat(3,downsamp_value_function_agent,downsamp_value_function);
end

% Plot mean state values, downsampled.
figure('Position',[500,500,250,250],'Color','w');
imagesc(nanmean(downsamp_value_function_agent,3),[0.8,1]);
xlabel('x (cm)');
ylabel('y (cm)')
xlim([0.5,10.5]);
ylim([0.5,10.5]);
axis square
ax = gca;
ax.Color = 'w';
ax.FontSize = 14;
ax.LineWidth = 1;
ax.XColor = 'k';
ax.YColor = 'k';
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;
ax.XTick = [0.5,5.5,10.5];
ax.XTickLabel = {'0','5','10'};
ax.YTick = [0.5,5.5,10.5];
ax.YTickLabel = {'10','5','0'};
colormap('redblue')

end