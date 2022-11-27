function get_tuning_curve_composite_task_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Get tuning curve.
% Input - Learning_stage: 'early' or 'late'.

for seed_num = 1:6
    clearvars -except learning_stage seed_num tuning_composite_task
    
    load('agent_behavior_composite_task.mat')
    load('agent_activity_composite_task.mat')
    
    behavior_composite_task_pretraining = behavior_composite_task.pretraining;
    activity_composite_task_pretraining = activity_composite_task.pretraining;
    
    if contains(learning_stage,'early')
        q_input = behavior_composite_task_pretraining{seed_num}.early.q_input;
        action = behavior_composite_task_pretraining{seed_num}.early.action;
        q_task1_network_first_layer = activity_composite_task_pretraining{seed_num}.early.q_task1_network_first_layer;
        q_task1_network_second_layer = activity_composite_task_pretraining{seed_num}.early.q_task1_network_second_layer;
        q_task1_network_third_layer = activity_composite_task_pretraining{seed_num}.early.q_task1_network_third_layer;
        q_task2_network_first_layer = activity_composite_task_pretraining{seed_num}.early.q_task2_network_first_layer;
        q_task2_network_second_layer = activity_composite_task_pretraining{seed_num}.early.q_task2_network_second_layer;
        q_task2_network_third_layer = activity_composite_task_pretraining{seed_num}.early.q_task2_network_third_layer;
        q_task1_network_first_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task1_network_first_layer_action_based;
        q_task1_network_second_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task1_network_second_layer_action_based;
        q_task1_network_third_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task1_network_third_layer_action_based;
        q_task2_network_first_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task2_network_first_layer_action_based;
        q_task2_network_second_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task2_network_second_layer_action_based;
        q_task2_network_third_layer_action_based = activity_composite_task_pretraining{seed_num}.early.q_task2_network_third_layer_action_based;
    elseif contains(learning_stage,'late')
        q_input = behavior_composite_task_pretraining{seed_num}.late.q_input;
        action = behavior_composite_task_pretraining{seed_num}.late.action;
        q_task1_network_first_layer = activity_composite_task_pretraining{seed_num}.late.q_task1_network_first_layer;
        q_task1_network_second_layer = activity_composite_task_pretraining{seed_num}.late.q_task1_network_second_layer;
        q_task1_network_third_layer = activity_composite_task_pretraining{seed_num}.late.q_task1_network_third_layer;
        q_task2_network_first_layer = activity_composite_task_pretraining{seed_num}.late.q_task2_network_first_layer;
        q_task2_network_second_layer = activity_composite_task_pretraining{seed_num}.late.q_task2_network_second_layer;
        q_task2_network_third_layer = activity_composite_task_pretraining{seed_num}.late.q_task2_network_third_layer;
        q_task1_network_first_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task1_network_first_layer_action_based;
        q_task1_network_second_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task1_network_second_layer_action_based;
        q_task1_network_third_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task1_network_third_layer_action_based;
        q_task2_network_first_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task2_network_first_layer_action_based;
        q_task2_network_second_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task2_network_second_layer_action_based;
        q_task2_network_third_layer_action_based = activity_composite_task_pretraining{seed_num}.late.q_task2_network_third_layer_action_based;
    end
    
    clear behavior_composite_task_pretraining activity_composite_task_pretraining % Save memory.
    
    % Space tuning.
    bin_size_pos = 40;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_task1_network_state{1}(:,x,y) = nanmean(q_task1_network_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_task1_network_state{2}(:,x,y) = nanmean(q_task1_network_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_task1_network_state{3}(:,x,y) = nanmean(q_task1_network_third_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_state{1}(:,x,y) = nanmean(q_task2_network_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_state{2}(:,x,y) = nanmean(q_task2_network_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_task2_network_state{3}(:,x,y) = nanmean(q_task2_network_third_layer(find(binX_pos == x & binY_pos == y),:));
        end
    end
    
    % Direction and lick tuning.
    q_task1_network_activity_action_based{1} = q_task1_network_first_layer_action_based;
    q_task1_network_activity_action_based{2} = q_task1_network_second_layer_action_based;
    q_task1_network_activity_action_based{3} = q_task1_network_third_layer_action_based;
    q_task2_network_activity_action_based{1} = q_task2_network_first_layer_action_based;
    q_task2_network_activity_action_based{2} = q_task2_network_second_layer_action_based;
    q_task2_network_activity_action_based{3} = q_task2_network_third_layer_action_based;
    
    action_concat = [];
    for trial_num = 1:length(action)
        action_concat = [action_concat;action{trial_num}];
    end
    
    for layer_num = 1:3
        q_task1_network_activity_action_based_concat{layer_num} = [];
        q_task2_network_activity_action_based_concat{layer_num} = [];
        for trial_num = 1:length(action)
            q_task1_network_activity_action_based_concat{layer_num} = [q_task1_network_activity_action_based_concat{layer_num},q_task1_network_activity_action_based{layer_num}{trial_num}'];
            q_task2_network_activity_action_based_concat{layer_num} = [q_task2_network_activity_action_based_concat{layer_num},q_task2_network_activity_action_based{layer_num}{trial_num}'];
        end
    end
    
    theta_temp = atan(abs(action_concat(:,2))./abs(action_concat(:,1)));
    for input_num = 1:size(action_concat,1)
        if action_concat(input_num,1) >= 0 && action_concat(input_num,2) >= 0
            theta(input_num) = theta_temp(input_num);
        elseif action_concat(input_num,1) < 0 && action_concat(input_num,2) >= 0
            theta(input_num) = pi - theta_temp(input_num);
        elseif action_concat(input_num,1) < 0 && action_concat(input_num,2) < 0
            theta(input_num) = pi + theta_temp(input_num);
        elseif action_concat(input_num,1) >= 0 && action_concat(input_num,2) < 0
            theta(input_num) = 2*pi - theta_temp(input_num);
        end
    end
    
    % Bin angle.
    [~,~,bin_angle] = histcounts(theta,[0:pi/8:2*pi]);
    
    % Combine bins.
    bin_angle_combined = zeros(1,size(action_concat,1));
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
    
    for layer_num = 1:3
        for theta_num = 1:8
            q_task1_network_dir{layer_num}(:,theta_num) = nanmean(q_task1_network_activity_action_based_concat{layer_num}(:,find(bin_angle_combined == theta_num)),2);
            q_task2_network_dir{layer_num}(:,theta_num) = nanmean(q_task2_network_activity_action_based_concat{layer_num}(:,find(bin_angle_combined == theta_num)),2);
        end
        q_task1_network_lick{layer_num}(:,1) = nanmean(q_task1_network_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) <= 0.08)),2);
        q_task1_network_lick{layer_num}(:,2) = nanmean(q_task1_network_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) > 0.08)),2);
        q_task2_network_lick{layer_num}(:,1) = nanmean(q_task2_network_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) <= 0.08)),2);
        q_task2_network_lick{layer_num}(:,2) = nanmean(q_task2_network_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) > 0.08)),2);
        % Obtain active cell index.
        q_task1_network_active_cell{layer_num} = find(sum(q_task1_network_activity_action_based_concat{layer_num},2) ~= 0);
        q_task2_network_active_cell{layer_num} = find(sum(q_task2_network_activity_action_based_concat{layer_num},2) ~= 0);
    end
    
    tuning_composite_task.pretraining{seed_num}.q_task1_network_state = q_task1_network_state;
    tuning_composite_task.pretraining{seed_num}.q_task2_network_state = q_task2_network_state;
    tuning_composite_task.pretraining{seed_num}.q_task1_network_dir = q_task1_network_dir;
    tuning_composite_task.pretraining{seed_num}.q_task2_network_dir = q_task2_network_dir;
    tuning_composite_task.pretraining{seed_num}.q_task1_network_lick = q_task1_network_lick;
    tuning_composite_task.pretraining{seed_num}.q_task2_network_lick = q_task2_network_lick;
    tuning_composite_task.pretraining{seed_num}.q_task1_network_active_cell = q_task1_network_active_cell;
    tuning_composite_task.pretraining{seed_num}.q_task2_network_active_cell = q_task2_network_active_cell;
end

end