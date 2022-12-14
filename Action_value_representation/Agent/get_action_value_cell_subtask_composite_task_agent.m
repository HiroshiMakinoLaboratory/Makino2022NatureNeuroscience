function get_action_value_cell_subtask_composite_task_agent

close all
clear all
clc

% Get subtask action value distribution for each neuron based on its activity in space.
% Load 'tuning_composite_task' after running 'get_tuning_curve_composite_task_agent.m' for the 'early' stage.

load('agent_behavior_task1.mat')
load('agent_behavior_task2.mat')
load('agent_activity_composite_task.mat')

for seed_num = 1:6
    clearvars -except behavior_task1 behavior_task2 activity_composite_task tuning_composite_task seed_num action_value_cell_subtask_composite_task
    
    q_input = activity_composite_task.pretraining{seed_num}.early.q_input;
    q_task1_network_first_layer = activity_composite_task.pretraining{seed_num}.early.q_task1_network_first_layer;
    q_task1_network_second_layer = activity_composite_task.pretraining{seed_num}.early.q_task1_network_second_layer;
    q_task1_network_third_layer = activity_composite_task.pretraining{seed_num}.early.q_task1_network_third_layer;
    q_task2_network_first_layer = activity_composite_task.pretraining{seed_num}.early.q_task2_network_first_layer;
    q_task2_network_second_layer = activity_composite_task.pretraining{seed_num}.early.q_task2_network_second_layer;
    q_task2_network_third_layer = activity_composite_task.pretraining{seed_num}.early.q_task2_network_third_layer;
    
    % Task1.
    observation = behavior_task1{seed_num}.late.observation;
    reward = behavior_task1{seed_num}.late.reward;
    
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
    clearvars -except behavior_task1 behavior_task2 activity_composite_task tuning_composite_task seed_num ...
        q_input q_task1_network_first_layer q_task1_network_second_layer q_task1_network_third_layer q_task2_network_first_layer q_task2_network_second_layer q_task2_network_third_layer ...
        gamma filtered_value_function action_value_cell_subtask_composite_task
    
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
    
    filtered_value_function_task1 = filtered_value_function;
    clearvars -except behavior_task1 behavior_task2 activity_composite_task tuning_composite_task seed_num ...
        q_input q_task1_network_first_layer q_task1_network_second_layer q_task1_network_third_layer q_task2_network_first_layer q_task2_network_second_layer q_task2_network_third_layer ...
        action_value_function filtered_value_function_task1 action_value_cell_subtask_composite_task
    
    % Task2.
    observation = behavior_task2{seed_num}.late.observation;
    reward = behavior_task2{seed_num}.late.reward;
    
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
    
    value_function = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
    image_filter = fspecial('gaussian',1,1);
    filtered_value_function = nanconv(value_function,image_filter,'edge','nanout');
    
    % Get action-value function for Task2.
    clearvars -except behavior_task1 behavior_task2 activity_composite_task tuning_composite_task seed_num ...
        q_input q_task1_network_first_layer q_task1_network_second_layer q_task1_network_third_layer q_task2_network_first_layer q_task2_network_second_layer q_task2_network_third_layer ...
        action_value_function filtered_value_function_task1 gamma filtered_value_function action_value_cell_subtask_composite_task
    
    % Downsample filtered_value_function.
    flipped_filtered_value_function = flipud(filtered_value_function); % Adjust for policy.
    downsamp_flipped_filtered_value_function_temp1 = cat(3,flipped_filtered_value_function(1:2:end,1:2:end),flipped_filtered_value_function(2:2:end,2:2:end));
    downsamp_flipped_filtered_value_function_temp2 = cat(3,flipped_filtered_value_function(1:2:end,2:2:end),flipped_filtered_value_function(2:2:end,1:2:end));
    downsamp_flipped_filtered_value_function = nanmean(cat(3,downsamp_flipped_filtered_value_function_temp1,downsamp_flipped_filtered_value_function_temp2),3); % Downsampling by averaging.
    
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            action_value_function{x_bin_num}{y_bin_num}(1,10:11) = nan; % 10 = no lick, 11 = lick.
            if x_bin_num == 5 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(10) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num);
                action_value_function{x_bin_num}{y_bin_num}(11) = 1;
            elseif x_bin_num == 6 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(10) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num);
                action_value_function{x_bin_num}{y_bin_num}(11) = 1;
            end
        end
    end
    
    filtered_value_function_task2 = filtered_value_function;
    % Convert nan to 0.
    filtered_value_function_task2(isnan(filtered_value_function_task2)) = 0;
    clearvars -except behavior_task1 behavior_task2 activity_composite_task tuning_composite_task seed_num ...
        q_input q_task1_network_first_layer q_task1_network_second_layer q_task1_network_third_layer q_task2_network_first_layer q_task2_network_second_layer q_task2_network_third_layer ...
        action_value_function filtered_value_function_task1 filtered_value_function_task2 action_value_cell_subtask_composite_task
    
    % Normalize.
    norm_filtered_value_function_task1 = filtered_value_function_task1./nanmax(filtered_value_function_task1(:));
    norm_filtered_value_function_task2 = filtered_value_function_task2./nanmax(filtered_value_function_task2(:));
    
    % Combine the two state-value functions.
    filtered_value_function = (norm_filtered_value_function_task1 + norm_filtered_value_function_task2)./2;
    
    % Identify states where neurons are active.
    q_task1_network_state = tuning_composite_task.pretraining{seed_num}.q_task1_network_state;
    q_task2_network_state = tuning_composite_task.pretraining{seed_num}.q_task2_network_state;
    q_task1_network_dir = tuning_composite_task.pretraining{seed_num}.q_task1_network_dir;
    q_task2_network_dir = tuning_composite_task.pretraining{seed_num}.q_task2_network_dir;
    q_task1_network_lick = tuning_composite_task.pretraining{seed_num}.q_task1_network_lick;
    q_task2_network_lick = tuning_composite_task.pretraining{seed_num}.q_task2_network_lick;
    
    % Determine action value distribution in active states for each neuron.
    for layer_num = 1:3
        for cell_num = 1:size(q_task1_network_state{layer_num},1)
            q1_task1_state_temp{layer_num}{cell_num} = squeeze(q_task1_network_state{layer_num}(cell_num,:,:));
            rotated_q1_task1_state{layer_num}{cell_num} = imrotate(q1_task1_state_temp{layer_num}{cell_num},90);
            flipped_q1_task1_state{layer_num}{cell_num} = flipud(rotated_q1_task1_state{layer_num}{cell_num}); % Adjust for action value.
            downsamp_flipped_q1_task1_state_temp1_1{layer_num}{cell_num} = cat(3,flipped_q1_task1_state{layer_num}{cell_num}(1:2:end,1:2:end),flipped_q1_task1_state{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q1_task1_state_temp1_2{layer_num}{cell_num} = cat(3,flipped_q1_task1_state{layer_num}{cell_num}(1:2:end,2:2:end),flipped_q1_task1_state{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q1_task1_state_temp{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q1_task1_state_temp1_1{layer_num}{cell_num},downsamp_flipped_q1_task1_state_temp1_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            downsamp_flipped_q1_task1_state_temp2_1{layer_num}{cell_num} = cat(3,downsamp_flipped_q1_task1_state_temp{layer_num}{cell_num}(1:2:end,1:2:end),downsamp_flipped_q1_task1_state_temp{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q1_task1_state_temp2_2{layer_num}{cell_num} = cat(3,downsamp_flipped_q1_task1_state_temp{layer_num}{cell_num}(1:2:end,2:2:end),downsamp_flipped_q1_task1_state_temp{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q1_task1_state{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q1_task1_state_temp2_1{layer_num}{cell_num},downsamp_flipped_q1_task1_state_temp2_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            norm_downsamp_flipped_q1_task1_state{layer_num}{cell_num} = (downsamp_flipped_q1_task1_state{layer_num}{cell_num} - min(min(downsamp_flipped_q1_task1_state{layer_num}{cell_num})))./(max(max(downsamp_flipped_q1_task1_state{layer_num}{cell_num} - min(min(downsamp_flipped_q1_task1_state{layer_num}{cell_num})))));
            [map_value_task1{layer_num}{cell_num},map_index_task1{layer_num}{cell_num}] = sort(norm_downsamp_flipped_q1_task1_state{layer_num}{cell_num}(:),'descend','MissingPlacement','last');
            
            action_dist_active_bins_all_movement_task1{layer_num}{cell_num} = [];
            action_dist_active_bins_all_lick_task1{layer_num}{cell_num} = [];
            if sum(map_value_task1{layer_num}{cell_num}(1:5)) >= 4.9999999999 || nansum(map_value_task1{layer_num}{cell_num}(1:5)) == 0 % If map values are all 1s or nans.
                action_dist_active_bins_all_movement_task1{layer_num}{cell_num} = [];
                action_dist_active_bins_all_lick_task1{layer_num}{cell_num} = [];
            else
                rows_map_value_temp_all_task1{layer_num} = [];
                cols_map_value_temp_all_task1{layer_num} = [];
                for bin_num = 1:5 % Top 5%.
                    [rows_map_value_temp_task1{layer_num}{cell_num}{bin_num},cols_map_value_temp_task1{layer_num}{cell_num}{bin_num}] = find(norm_downsamp_flipped_q1_task1_state{layer_num}{cell_num} == map_value_task1{layer_num}{cell_num}(bin_num));
                    rows_map_value_temp_all_task1{layer_num} = [rows_map_value_temp_all_task1{layer_num};rows_map_value_temp_task1{layer_num}{cell_num}{bin_num}];
                    cols_map_value_temp_all_task1{layer_num} = [cols_map_value_temp_all_task1{layer_num};cols_map_value_temp_task1{layer_num}{cell_num}{bin_num}];
                end
                rows_map_value_task1{layer_num}{cell_num} = rows_map_value_temp_all_task1{layer_num}(1:5);
                cols_map_value_task1{layer_num}{cell_num} = cols_map_value_temp_all_task1{layer_num}(1:5);
                for bin_num = 1:5 % Top 5%.
                    action_dist_active_bins_movement_task1{layer_num}{cell_num}{bin_num} = action_value_function{cols_map_value_task1{layer_num}{cell_num}(bin_num)}{rows_map_value_task1{layer_num}{cell_num}(bin_num)}(1:9);
                    action_dist_active_bins_movement_task1{layer_num}{cell_num}{bin_num} = action_dist_active_bins_movement_task1{layer_num}{cell_num}{bin_num}.*map_value_task1{layer_num}{cell_num}(bin_num); % Weight.
                    action_dist_active_bins_all_movement_task1{layer_num}{cell_num} = [action_dist_active_bins_all_movement_task1{layer_num}{cell_num};action_dist_active_bins_movement_task1{layer_num}{cell_num}{bin_num}];
                    
                    action_dist_active_bins_lick_task1{layer_num}{cell_num}{bin_num} = action_value_function{cols_map_value_task1{layer_num}{cell_num}(bin_num)}{rows_map_value_task1{layer_num}{cell_num}(bin_num)}(10:11);
                    action_dist_active_bins_lick_task1{layer_num}{cell_num}{bin_num} = action_dist_active_bins_lick_task1{layer_num}{cell_num}{bin_num}.*map_value_task1{layer_num}{cell_num}(bin_num); % Weight.
                    action_dist_active_bins_all_lick_task1{layer_num}{cell_num} = [action_dist_active_bins_all_lick_task1{layer_num}{cell_num};action_dist_active_bins_lick_task1{layer_num}{cell_num}{bin_num}];
                end
            end
            
            if ~isempty(action_dist_active_bins_all_movement_task1{layer_num}{cell_num})
                mean_action_dist_active_bins_all_movement_task1{layer_num}{cell_num} = nanmean(action_dist_active_bins_all_movement_task1{layer_num}{cell_num});
            else
                mean_action_dist_active_bins_all_movement_task1{layer_num}{cell_num} = nan(1,9);
            end
            
            if ~isempty(action_dist_active_bins_all_lick_task1{layer_num}{cell_num})
                mean_action_dist_active_bins_all_lick_task1{layer_num}{cell_num} = nanmean(action_dist_active_bins_all_lick_task1{layer_num}{cell_num});
            else
                mean_action_dist_active_bins_all_lick_task1{layer_num}{cell_num} = nan(1,2);
            end
        end
        
        for cell_num = 1:size(q_task2_network_state{layer_num},1)
            q1_task2_state_temp{layer_num}{cell_num} = squeeze(q_task2_network_state{layer_num}(cell_num,:,:));
            rotated_q1_task2_state{layer_num}{cell_num} = imrotate(q1_task2_state_temp{layer_num}{cell_num},90);
            flipped_q1_task2_state{layer_num}{cell_num} = flipud(rotated_q1_task2_state{layer_num}{cell_num}); % Adjust for action-value.
            downsamp_flipped_q1_task2_state_temp1_1{layer_num}{cell_num} = cat(3,flipped_q1_task2_state{layer_num}{cell_num}(1:2:end,1:2:end),flipped_q1_task2_state{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q1_task2_state_temp1_2{layer_num}{cell_num} = cat(3,flipped_q1_task2_state{layer_num}{cell_num}(1:2:end,2:2:end),flipped_q1_task2_state{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q1_task2_state_temp{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q1_task2_state_temp1_1{layer_num}{cell_num},downsamp_flipped_q1_task2_state_temp1_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            downsamp_flipped_q1_task2_state_temp2_1{layer_num}{cell_num} = cat(3,downsamp_flipped_q1_task2_state_temp{layer_num}{cell_num}(1:2:end,1:2:end),downsamp_flipped_q1_task2_state_temp{layer_num}{cell_num}(2:2:end,2:2:end));
            downsamp_flipped_q1_task2_state_temp2_2{layer_num}{cell_num} = cat(3,downsamp_flipped_q1_task2_state_temp{layer_num}{cell_num}(1:2:end,2:2:end),downsamp_flipped_q1_task2_state_temp{layer_num}{cell_num}(2:2:end,1:2:end));
            downsamp_flipped_q1_task2_state{layer_num}{cell_num} = nanmean(cat(3,downsamp_flipped_q1_task2_state_temp2_1{layer_num}{cell_num},downsamp_flipped_q1_task2_state_temp2_2{layer_num}{cell_num}),3); % Downsampling by averaging.
            norm_downsamp_flipped_q1_task2_state{layer_num}{cell_num} = (downsamp_flipped_q1_task2_state{layer_num}{cell_num} - min(min(downsamp_flipped_q1_task2_state{layer_num}{cell_num})))./(max(max(downsamp_flipped_q1_task2_state{layer_num}{cell_num} - min(min(downsamp_flipped_q1_task2_state{layer_num}{cell_num})))));
            [map_value_task2{layer_num}{cell_num},map_index_task2{layer_num}{cell_num}] = sort(norm_downsamp_flipped_q1_task2_state{layer_num}{cell_num}(:),'descend','MissingPlacement','last');
            
            action_dist_active_bins_all_movement_task2{layer_num}{cell_num} = [];
            action_dist_active_bins_all_lick_task2{layer_num}{cell_num} = [];
            if sum(map_value_task2{layer_num}{cell_num}(1:5)) >= 4.9999999999 || nansum(map_value_task2{layer_num}{cell_num}(1:5)) == 0 % If map values are all 1 or nan.
                action_dist_active_bins_all_movement_task2{layer_num}{cell_num} = [];
                action_dist_active_bins_all_lick_task2{layer_num}{cell_num} = [];
            else
                rows_map_value_temp_all_task2{layer_num} = [];
                cols_map_value_temp_all_task2{layer_num} = [];
                for bin_num = 1:5 % Top 5%.
                    [rows_map_value_temp_task2{layer_num}{cell_num}{bin_num},cols_map_value_temp_task2{layer_num}{cell_num}{bin_num}] = find(norm_downsamp_flipped_q1_task2_state{layer_num}{cell_num} == map_value_task2{layer_num}{cell_num}(bin_num));
                    rows_map_value_temp_all_task2{layer_num} = [rows_map_value_temp_all_task2{layer_num};rows_map_value_temp_task2{layer_num}{cell_num}{bin_num}];
                    cols_map_value_temp_all_task2{layer_num} = [cols_map_value_temp_all_task2{layer_num};cols_map_value_temp_task2{layer_num}{cell_num}{bin_num}];
                end
                rows_map_value_task2{layer_num}{cell_num} = rows_map_value_temp_all_task2{layer_num}(1:5);
                cols_map_value_task2{layer_num}{cell_num} = cols_map_value_temp_all_task2{layer_num}(1:5);
                for bin_num = 1:5 % Top 5%.
                    action_dist_active_bins_movement_task2{layer_num}{cell_num}{bin_num} = action_value_function{cols_map_value_task2{layer_num}{cell_num}(bin_num)}{rows_map_value_task2{layer_num}{cell_num}(bin_num)}(1:9);
                    action_dist_active_bins_movement_task2{layer_num}{cell_num}{bin_num} = action_dist_active_bins_movement_task2{layer_num}{cell_num}{bin_num}.*map_value_task2{layer_num}{cell_num}(bin_num); % Weight.
                    action_dist_active_bins_all_movement_task2{layer_num}{cell_num} = [action_dist_active_bins_all_movement_task2{layer_num}{cell_num};action_dist_active_bins_movement_task2{layer_num}{cell_num}{bin_num}];
                    
                    action_dist_active_bins_lick_task2{layer_num}{cell_num}{bin_num} = action_value_function{cols_map_value_task2{layer_num}{cell_num}(bin_num)}{rows_map_value_task2{layer_num}{cell_num}(bin_num)}(10:11);
                    action_dist_active_bins_lick_task2{layer_num}{cell_num}{bin_num} = action_dist_active_bins_lick_task2{layer_num}{cell_num}{bin_num}.*map_value_task2{layer_num}{cell_num}(bin_num); % Weight.
                    action_dist_active_bins_all_lick_task2{layer_num}{cell_num} = [action_dist_active_bins_all_lick_task2{layer_num}{cell_num};action_dist_active_bins_lick_task2{layer_num}{cell_num}{bin_num}];
                end
            end
            
            if ~isempty(action_dist_active_bins_all_movement_task2{layer_num}{cell_num})
                mean_action_dist_active_bins_all_movement_task2{layer_num}{cell_num} = nanmean(action_dist_active_bins_all_movement_task2{layer_num}{cell_num});
            else
                mean_action_dist_active_bins_all_movement_task2{layer_num}{cell_num} = nan(1,9);
            end
            
            if ~isempty(action_dist_active_bins_all_lick_task2{layer_num}{cell_num})
                mean_action_dist_active_bins_all_lick_task2{layer_num}{cell_num} = nanmean(action_dist_active_bins_all_lick_task2{layer_num}{cell_num});
            else
                mean_action_dist_active_bins_all_lick_task2{layer_num}{cell_num} = nan(1,2);
            end
        end
    end
    
    % Metric.
    for layer_num = 1:3
        for cell_num = 1:length(mean_action_dist_active_bins_all_movement_task1{layer_num})
            correlation_movement_task1{layer_num}(cell_num) = corr([q_task1_network_dir{layer_num}(cell_num,:),nan]',mean_action_dist_active_bins_all_movement_task1{layer_num}{cell_num}','rows','complete'); % Nan for no movement, which was not present in the agent.
            dot_product_movement_task1{layer_num}(cell_num) = nansum([q_task1_network_dir{layer_num}(cell_num,:),nan].*mean_action_dist_active_bins_all_movement_task1{layer_num}{cell_num}); % Nan for no movement, which was not present in the agent.
        end
        for cell_num = 1:length(mean_action_dist_active_bins_all_movement_task2{layer_num})
            correlation_movement_task2{layer_num}(cell_num) = corr([q_task2_network_dir{layer_num}(cell_num,:),nan]',mean_action_dist_active_bins_all_movement_task2{layer_num}{cell_num}','rows','complete'); % Nan for no movement, which was not present in the agent.
            dot_product_movement_task2{layer_num}(cell_num) = nansum([q_task2_network_dir{layer_num}(cell_num,:),nan].*mean_action_dist_active_bins_all_movement_task2{layer_num}{cell_num}); % Nan for no movement, which was not present in the agent.
        end
        
        for cell_num = 1:length(mean_action_dist_active_bins_all_lick_task1{layer_num})
            dot_product_lick_task1{layer_num}(cell_num) = nansum(q_task1_network_lick{layer_num}(cell_num,:).*mean_action_dist_active_bins_all_lick_task1{layer_num}{cell_num});
            correlation_lick_task1{layer_num}(cell_num) = corr(q_task1_network_lick{layer_num}(cell_num,:)',mean_action_dist_active_bins_all_lick_task1{layer_num}{cell_num}','rows','complete');
        end
        for cell_num = 1:length(mean_action_dist_active_bins_all_lick_task2{layer_num})
            dot_product_lick_task2{layer_num}(cell_num) = nansum(q_task2_network_lick{layer_num}(cell_num,:).*mean_action_dist_active_bins_all_lick_task2{layer_num}{cell_num});
            correlation_lick_task2{layer_num}(cell_num) = corr(q_task2_network_lick{layer_num}(cell_num,:)',mean_action_dist_active_bins_all_lick_task2{layer_num}{cell_num}','rows','complete');
        end
    end
    
    bin_size_pos = 20;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    % Obtain space tuning (20x20).
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_task1_network_first_layer_state_20x20(:,x,y) = nanmean(q_task1_network_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_task1_network_second_layer_state_20x20(:,x,y) = nanmean(q_task1_network_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_task1_network_third_layer_state_20x20(:,x,y) = nanmean(q_task1_network_third_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_first_layer_state_20x20(:,x,y) = nanmean(q_task2_network_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_second_layer_state_20x20(:,x,y) = nanmean(q_task2_network_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_third_layer_state_20x20(:,x,y) = nanmean(q_task2_network_third_layer(find(binX_pos == x & binY_pos == y),:));
        end
    end
    
    % Filter and rotate.
    for cell_num = 1:size(q_task1_network_first_layer_state_20x20,1)
        q_task1_network_space_tuning{1}{cell_num} = imrotate(imgaussfilt(squeeze(q_task1_network_first_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_task1_network_second_layer_state_20x20,1)
        q_task1_network_space_tuning{2}{cell_num} = imrotate(imgaussfilt(squeeze(q_task1_network_second_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_task1_network_third_layer_state_20x20,1)
        q_task1_network_space_tuning{3}{cell_num} = imrotate(imgaussfilt(squeeze(q_task1_network_third_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_task2_network_first_layer_state_20x20,1)
        q_task2_network_space_tuning{1}{cell_num} = imrotate(imgaussfilt(squeeze(q_task2_network_first_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_task2_network_second_layer_state_20x20,1)
        q_task2_network_space_tuning{2}{cell_num} = imrotate(imgaussfilt(squeeze(q_task2_network_second_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    for cell_num = 1:size(q_task2_network_third_layer_state_20x20,1)
        q_task2_network_space_tuning{3}{cell_num} = imrotate(imgaussfilt(squeeze(q_task2_network_third_layer_state_20x20(cell_num,:,:)),[1,1]),90);
    end
    
    % Metric.
    for layer_num = 1:3
        for cell_num = 1:length(q_task1_network_space_tuning{layer_num})
            correlation_agent_position_task1{layer_num}(cell_num) = corr(q_task1_network_space_tuning{layer_num}{cell_num}(:),filtered_value_function(:),'rows','complete');
        end
        for cell_num = 1:length(q_task2_network_space_tuning{layer_num})
            correlation_agent_position_task2{layer_num}(cell_num) = corr(q_task2_network_space_tuning{layer_num}{cell_num}(:),filtered_value_function(:),'rows','complete');
        end
    end
    
    action_value_cell_subtask_composite_task{seed_num}.action_value_function = action_value_function;
    action_value_cell_subtask_composite_task{seed_num}.movement_task1_network = mean_action_dist_active_bins_all_movement_task1;
    action_value_cell_subtask_composite_task{seed_num}.movement_task2_network = mean_action_dist_active_bins_all_movement_task2;
    action_value_cell_subtask_composite_task{seed_num}.lick_task1_network = mean_action_dist_active_bins_all_lick_task1;
    action_value_cell_subtask_composite_task{seed_num}.lick_task2_network = mean_action_dist_active_bins_all_lick_task2;
    action_value_cell_subtask_composite_task{seed_num}.correlation_agent_movement_task1 = correlation_movement_task1;
    action_value_cell_subtask_composite_task{seed_num}.dot_product_agent_movement_task1 = dot_product_movement_task1;
    action_value_cell_subtask_composite_task{seed_num}.correlation_agent_movement_task2 = correlation_movement_task2;
    action_value_cell_subtask_composite_task{seed_num}.dot_product_agent_movement_task2 = dot_product_movement_task2;
    action_value_cell_subtask_composite_task{seed_num}.dot_product_lick_task1 = dot_product_lick_task1;
    action_value_cell_subtask_composite_task{seed_num}.correlation_lick_task1 = correlation_lick_task1;
    action_value_cell_subtask_composite_task{seed_num}.dot_product_lick_task2 = dot_product_lick_task2;
    action_value_cell_subtask_composite_task{seed_num}.correlation_lick_task2 = correlation_lick_task2;
    action_value_cell_subtask_composite_task{seed_num}.correlation_agent_position_task1 = correlation_agent_position_task1;
    action_value_cell_subtask_composite_task{seed_num}.correlation_agent_position_task2 = correlation_agent_position_task2;
end

end