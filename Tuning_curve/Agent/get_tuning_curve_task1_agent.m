function get_tuning_curve_task1_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Get tuning curve.
% Input - Learning_stage: 'early', 'late' or 'late_deterministic'.

for seed_num = 1:6
    clearvars -except learning_stage seed_num tuning_task1
    
    load('agent_behavior_task1.mat')
    load('agent_activity_task1.mat')
    
    if contains(learning_stage,'early')
        q_input = behavior_task1{seed_num}.early.q_input;
        action = behavior_task1{seed_num}.early.action;
        q_first_layer = activity_task1{seed_num}.early.q_first_layer;
        q_second_layer = activity_task1{seed_num}.early.q_second_layer;
        q_third_layer = activity_task1{seed_num}.early.q_third_layer;
        q_first_layer_action_based = activity_task1{seed_num}.early.q_first_layer_action_based;
        q_second_layer_action_based = activity_task1{seed_num}.early.q_second_layer_action_based;
        q_third_layer_action_based = activity_task1{seed_num}.early.q_third_layer_action_based;
    elseif contains(learning_stage,'late_deterministic')
        q_input = behavior_task1{seed_num}.late_deterministic.q_input;
        action = behavior_task1{seed_num}.late_deterministic.action;
        q_first_layer = activity_task1{seed_num}.late_deterministic.q_first_layer;
        q_second_layer = activity_task1{seed_num}.late_deterministic.q_second_layer;
        q_third_layer = activity_task1{seed_num}.late_deterministic.q_third_layer;
        q_first_layer_action_based = activity_task1{seed_num}.late_deterministic.q_first_layer_action_based;
        q_second_layer_action_based = activity_task1{seed_num}.late_deterministic.q_second_layer_action_based;
        q_third_layer_action_based = activity_task1{seed_num}.late_deterministic.q_third_layer_action_based;
    elseif contains(learning_stage,'late')
        q_input = behavior_task1{seed_num}.late.q_input;
        action = behavior_task1{seed_num}.late.action;
        q_first_layer = activity_task1{seed_num}.late.q_first_layer;
        q_second_layer = activity_task1{seed_num}.late.q_second_layer;
        q_third_layer = activity_task1{seed_num}.late.q_third_layer;
        q_first_layer_action_based = activity_task1{seed_num}.late.q_first_layer_action_based;
        q_second_layer_action_based = activity_task1{seed_num}.late.q_second_layer_action_based;
        q_third_layer_action_based = activity_task1{seed_num}.late.q_third_layer_action_based;
    end
    
    clear behavior_task1 activity_task1 % Save memory.
    
    % Space tuning.
    bin_size_pos = 40;
    [~,~,~,binX_pos,binY_pos] = histcounts2(q_input(:,1),q_input(:,2),bin_size_pos); % x and y position.
    
    for x = 1:bin_size_pos
        for y = 1:bin_size_pos
            q_state{1}(:,x,y) = nanmean(q_first_layer(find(binX_pos == x & binY_pos == y),:));
            q_state{2}(:,x,y) = nanmean(q_second_layer(find(binX_pos == x & binY_pos == y),:));
            q_state{3}(:,x,y) = nanmean(q_third_layer(find(binX_pos == x & binY_pos == y),:));
        end
    end
    
    % Direction tuning.
    q_activity_action_based{1} = q_first_layer_action_based;
    q_activity_action_based{2} = q_second_layer_action_based;
    q_activity_action_based{3} = q_third_layer_action_based;
    
    action_concat = [];
    for trial_num = 1:length(action)
        action_concat = [action_concat;action{trial_num}];
    end
    
    for layer_num = 1:3
        q_activity_action_based_concat{layer_num} = [];
        for trial_num = 1:length(action)
            q_activity_action_based_concat{layer_num} = [q_activity_action_based_concat{layer_num},q_activity_action_based{layer_num}{trial_num}'];
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
            q_dir{layer_num}(:,theta_num) = nanmean(q_activity_action_based_concat{layer_num}(:,find(bin_angle_combined == theta_num)),2);
        end
        % Obtain active cell index.
        q_active_cell{layer_num} = find(sum(q_activity_action_based_concat{layer_num},2) ~= 0);
    end
    
    tuning_task1{seed_num}.q_state = q_state;
    tuning_task1{seed_num}.q_dir = q_dir;
    tuning_task1{seed_num}.q_active_cell = q_active_cell;
end

end