function get_action_value_cell_task1_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Get action value distribution for each neuron based on its activity in space.
% Input - Learning_stage: 'early' or 'late'.
% Load 'tuning_task1' after running 'get_tuning_curve_task1_agent.m'.

load('agent_behavior_task1.mat')
load('agent_activity_task1.mat')

for seed_num = 1:6
    clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 seed_num action_value_cell_task1
    
    if contains(learning_stage,'early')
        observation = behavior_task1{seed_num}.early.observation;
        reward = behavior_task1{seed_num}.early.reward;
        q_input = activity_task1{seed_num}.early.q_input;
        q_first_layer = activity_task1{seed_num}.early.q_first_layer;
        q_second_layer = activity_task1{seed_num}.early.q_second_layer;
        q_third_layer = activity_task1{seed_num}.early.q_third_layer;
    elseif contains(learning_stage,'late')
        observation = behavior_task1{seed_num}.late.observation;
        reward = behavior_task1{seed_num}.late.reward;
        q_input = activity_task1{seed_num}.late.q_input;
        q_first_layer = activity_task1{seed_num}.late.q_first_layer;
        q_second_layer = activity_task1{seed_num}.late.q_second_layer;
        q_third_layer = activity_task1{seed_num}.late.q_third_layer;
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
        mean_step_size_from_state(trial_num,7:14,7:14) = 1;
    end
    
    % Incorporate miss trials.
    if ~isempty(incorrect_trial) == 1
        for incorrect_trial_num = 1:length(incorrect_trial)
            mean_step_size_from_state(incorrect_trial(incorrect_trial_num),:,:) = zeros(1,20,20);
        end
    end
    
    value_function = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
    image_filter = fspecial('gaussian',1,1);
    filtered_value_function = nanconv(value_function,image_filter,'edge','nanout');
    
    % Get action-value function for Task1.
    clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 seed_num ...
        q_input q_first_layer q_second_layer q_third_layer ...
        gamma filtered_value_function action_value_cell_task1
    
    % Downsample filtered_value_function.
    flipped_filtered_value_function = flipud(filtered_value_function); % Adjust for policy.
    downsamp_flipped_filtered_value_function_temp1 = cat(3,flipped_filtered_value_function(1:2:end,1:2:end),flipped_filtered_value_function(2:2:end,2:2:end));
    downsamp_flipped_filtered_value_function_temp2 = cat(3,flipped_filtered_value_function(1:2:end,2:2:end),flipped_filtered_value_function(2:2:end,1:2:end));
    downsamp_flipped_filtered_value_function = nanmean(cat(3,downsamp_flipped_filtered_value_function_temp1,downsamp_flipped_filtered_value_function_temp2),3); % Downsampling by averaging.
    downsamp_flipped_filtered_value_function(4:7,4:7) = 1; % 1 for the reward zone.
    
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            action_value_function{x_bin_num}{y_bin_num} = nan(1,9);
            if x_bin_num == 1 && y_bin_num == 1 % Bottom-left of the real coordinate.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 1 && y_bin_num == 10 % Top-left of the real coordinate.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 10 && y_bin_num == 1 % Bottom-right of the real coordinate.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 10 && y_bin_num == 10 % Top-right of the real coordinate.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 1 % Left edge.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 10 % Right edge.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif y_bin_num == 1 % Bottom edge.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif y_bin_num == 10 % Top edge.
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 4
                action_value_function{x_bin_num}{y_bin_num}(1) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 5
                action_value_function{x_bin_num}{y_bin_num}(1) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 6
                action_value_function{x_bin_num}{y_bin_num}(1) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 7
                action_value_function{x_bin_num}{y_bin_num}(1) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 3 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 4
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 5
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 6
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 7
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 8 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 4 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 5 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 6 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 7 && y_bin_num == 3
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 4 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 5 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 6 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            elseif x_bin_num == 7 && y_bin_num == 8
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
                
            else
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num + 1); % East.
                action_value_function{x_bin_num}{y_bin_num}(2) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num + 1); % Northeast.
                action_value_function{x_bin_num}{y_bin_num}(3) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num); % North.
                action_value_function{x_bin_num}{y_bin_num}(4) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num + 1,x_bin_num - 1); % Northwest.
                action_value_function{x_bin_num}{y_bin_num}(5) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % West.
                action_value_function{x_bin_num}{y_bin_num}(6) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num - 1); % Southwest.
                action_value_function{x_bin_num}{y_bin_num}(7) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num); % South.
                action_value_function{x_bin_num}{y_bin_num}(8) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num - 1,x_bin_num + 1); % Southeast.
                action_value_function{x_bin_num}{y_bin_num}(9) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % No movement.
            end
        end
    end
    
    clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 seed_num ...
        q_input q_first_layer q_second_layer q_third_layer ...
        filtered_value_function action_value_function action_value_cell_task1
    
    % Identify states where neurons are active.
    q_state = tuning_task1{seed_num}.q_state;
    q_dir = tuning_task1{seed_num}.q_dir;
    
    % Determine action value distribution in active states for each neuron.
    for layer_num = 1:3
        for cell_num = 1:size(q_state{layer_num},1)
            q_state_temp{layer_num}{cell_num} = squeeze(q_state{layer_num}(cell_num,:,:));
            rotated_q_state{layer_num}{cell_num} = imrotate(q_state_temp{layer_num}{cell_num},90);
            flipped_q_state{layer_num}{cell_num} = flipud(rotated_q_state{layer_num}{cell_num}); % Adjust for action value.
            downsamp_flipped_q_state_temp1_1{layer_num}{cell_num} = cat(3,flipped_q_state{layer_num}{cell_num}(1:2:end,1:2:end),flipped_q_state{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q_state_temp1_2{layer_num}{cell_num} = cat(3,flipped_q_state{layer_num}{cell_num}(1:2:end,2:2:end),flipped_q_state{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q_state_temp{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q_state_temp1_1{layer_num}{cell_num},downsamp_flipped_q_state_temp1_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            downsamp_flipped_q_state_temp2_1{layer_num}{cell_num} = cat(3,downsamp_flipped_q_state_temp{layer_num}{cell_num}(1:2:end,1:2:end),downsamp_flipped_q_state_temp{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q_state_temp2_2{layer_num}{cell_num} = cat(3,downsamp_flipped_q_state_temp{layer_num}{cell_num}(1:2:end,2:2:end),downsamp_flipped_q_state_temp{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q_state{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q_state_temp2_1{layer_num}{cell_num},downsamp_flipped_q_state_temp2_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            norm_downsamp_flipped_q_state{layer_num}{cell_num} = (downsamp_flipped_q_state{layer_num}{cell_num} - min(min(downsamp_flipped_q_state{layer_num}{cell_num})))./(max(max(downsamp_flipped_q_state{layer_num}{cell_num} - min(min(downsamp_flipped_q_state{layer_num}{cell_num})))));
            [map_value{layer_num}{cell_num},map_index{layer_num}{cell_num}] = sort(norm_downsamp_flipped_q_state{layer_num}{cell_num}(:),'descend','MissingPlacement','last');
            
            action_dist_active_bins_all_movement{layer_num}{cell_num} = [];
            if sum(map_value{layer_num}{cell_num}(1:5)) >= 4.9999999999 || nansum(map_value{layer_num}{cell_num}(1:5)) == 0 % If map values are all 1s or nans.
                action_dist_active_bins_all_movement{layer_num}{cell_num} = [];
            else
                rows_map_value_temp_all{layer_num} = [];
                cols_map_value_temp_all{layer_num} = [];
                for bin_num = 1:5 % Top 5%.
                    [rows_map_value_temp{layer_num}{cell_num}{bin_num},cols_map_value_temp{layer_num}{cell_num}{bin_num}] = find(norm_downsamp_flipped_q_state{layer_num}{cell_num} == map_value{layer_num}{cell_num}(bin_num));
                    rows_map_value_temp_all{layer_num} = [rows_map_value_temp_all{layer_num};rows_map_value_temp{layer_num}{cell_num}{bin_num}];
                    cols_map_value_temp_all{layer_num} = [cols_map_value_temp_all{layer_num};cols_map_value_temp{layer_num}{cell_num}{bin_num}];
                end
                rows_map_value{layer_num}{cell_num} = rows_map_value_temp_all{layer_num}(1:5);
                cols_map_value{layer_num}{cell_num} = cols_map_value_temp_all{layer_num}(1:5);
                for bin_num = 1:5 % Top 5%.
                    action_dist_active_bins_movement{layer_num}{cell_num}{bin_num} = action_value_function{cols_map_value{layer_num}{cell_num}(bin_num)}{rows_map_value{layer_num}{cell_num}(bin_num)}(1:9);
                    action_dist_active_bins_movement{layer_num}{cell_num}{bin_num} = action_dist_active_bins_movement{layer_num}{cell_num}{bin_num}.*map_value{layer_num}{cell_num}(bin_num); % Weight.
                    action_dist_active_bins_all_movement{layer_num}{cell_num} = [action_dist_active_bins_all_movement{layer_num}{cell_num};action_dist_active_bins_movement{layer_num}{cell_num}{bin_num}];
                end
            end
            
            if ~isempty(action_dist_active_bins_all_movement{layer_num}{cell_num})
                mean_action_dist_active_bins_all_movement{layer_num}{cell_num} = nanmean(action_dist_active_bins_all_movement{layer_num}{cell_num});
            else
                mean_action_dist_active_bins_all_movement{layer_num}{cell_num} = nan(1,9);
            end
        end
    end
    
    % Metric.
    for layer_num = 1:3
        for cell_num = 1:length(mean_action_dist_active_bins_all_movement{layer_num})
            dot_product_agent_movement{layer_num}(cell_num) = nansum([q_dir{layer_num}(cell_num,:),nan].*mean_action_dist_active_bins_all_movement{layer_num}{cell_num}); % Nan for no movement, which was not present in the agent.
            correlation_agent_movement{layer_num}(cell_num) = corr([q_dir{layer_num}(cell_num,:),nan]',mean_action_dist_active_bins_all_movement{layer_num}{cell_num}','rows','complete'); % Nan for no movement, which was not present in the agent.
        end
    end
    
    bin_size_pos = 20;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    % Obtain space tuning (20x20).
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_first_layer_state_20x20(:,x,y) = nanmean(q_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_second_layer_state_20x20(:,x,y) = nanmean(q_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_third_layer_state_20x20(:,x,y) = nanmean(q_third_layer(find(binX_pos == x & binY_pos == y),:));
        end
    end
    
    % Filter and rotate.
    for cell_num = 1:size(q_first_layer_state_20x20,1)
        q_space_tuning{1}{cell_num} = imrotate(imgaussfilt(squeeze(q_first_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_second_layer_state_20x20,1)
        q_space_tuning{2}{cell_num} = imrotate(imgaussfilt(squeeze(q_second_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_third_layer_state_20x20,1)
        q_space_tuning{3}{cell_num} = imrotate(imgaussfilt(squeeze(q_third_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    
    % Metric.
    for layer_num = 1:3
        for cell_num = 1:length(q_space_tuning{layer_num})
            correlation_agent_position{layer_num}(cell_num) = corr(q_space_tuning{layer_num}{cell_num}(:),filtered_value_function(:),'rows','complete');
        end
    end
    
    action_value_cell_task1{seed_num}.action_value_function = action_value_function;
    action_value_cell_task1{seed_num}.action_value_dist = mean_action_dist_active_bins_all_movement;
    action_value_cell_task1{seed_num}.dot_product_agent_movement = dot_product_agent_movement;
    action_value_cell_task1{seed_num}.correlation_agent_movement = correlation_agent_movement;
    action_value_cell_task1{seed_num}.correlation_agent_position = correlation_agent_position;
end

end