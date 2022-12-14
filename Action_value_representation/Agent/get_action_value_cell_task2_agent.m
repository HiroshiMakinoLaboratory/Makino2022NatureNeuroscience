function get_action_value_cell_task2_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Get action value distribution for each neuron.
% Input - Learning_stage: 'early' or 'late'.
% Load 'tuning_task2' after running 'get_tuning_curve_task2_agent.m'.

load('agent_behavior_task2.mat')

for seed_num = 1:6
    clearvars -except learning_stage behavior_task2 tuning_task2 seed_num action_value_cell_task2
    
    if contains(learning_stage,'early')
        observation = behavior_task2{seed_num}.early.observation;
        reward = behavior_task2{seed_num}.early.reward;
    elseif contains(learning_stage,'late')
        observation = behavior_task2{seed_num}.late.observation;
        reward = behavior_task2{seed_num}.late.reward;
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
    
    % Rotate and filter.
    value_function = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
    image_filter = fspecial('gaussian',1,1);
    filtered_value_function = nanconv(value_function,image_filter,'edge','nanout');
    
    % Get action-value function for Task2.
    clearvars -except learning_stage behavior_task2 tuning_task2 seed_num gamma filtered_value_function action_value_cell_task2
    
    % Downsample filtered_value_function.
    flipped_filtered_value_function = flipud(filtered_value_function); % Adjust for policy.
    downsamp_flipped_filtered_value_function_temp1 = cat(3,flipped_filtered_value_function(1:2:end,1:2:end),flipped_filtered_value_function(2:2:end,2:2:end));
    downsamp_flipped_filtered_value_function_temp2 = cat(3,flipped_filtered_value_function(1:2:end,2:2:end),flipped_filtered_value_function(2:2:end,1:2:end));
    downsamp_flipped_filtered_value_function = nanmean(cat(3,downsamp_flipped_filtered_value_function_temp1,downsamp_flipped_filtered_value_function_temp2),3); % Downsampling by averaging.
    
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            action_value_function{x_bin_num}{y_bin_num}(1,1:2) = nan; % 1 = no lick, 2 = lick.
            if x_bin_num == 5 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num);
                action_value_function{x_bin_num}{y_bin_num}(2) = 1;
            elseif x_bin_num == 6 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num);
                action_value_function{x_bin_num}{y_bin_num}(2) = 1;
            end
        end
    end
    action_value_function_lick = (action_value_function{5}{1} + action_value_function{6}{1})./2;
    
    q_lick = tuning_task2{seed_num}.q_lick;
    
    % Metric.
    for layer_num = 1:3
        for cell_num = 1:size(q_lick{layer_num},1)
            dot_product_lick{layer_num}(cell_num) = nansum(q_lick{layer_num}(cell_num,:).*action_value_function_lick);
            correlation_lick{layer_num}(cell_num) = corr(q_lick{layer_num}(cell_num,:)',action_value_function_lick','rows','complete');
        end
    end
    
    action_value_cell_task2{seed_num}.action_value_function = action_value_function;
    action_value_cell_task2{seed_num}.action_value_dist = action_value_function_lick;
    action_value_cell_task2{seed_num}.dot_product_lick = dot_product_lick;
    action_value_cell_task2{seed_num}.correlation_lick = correlation_lick;
end

end