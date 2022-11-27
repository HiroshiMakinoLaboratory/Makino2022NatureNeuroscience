function get_tuning_curve_task2_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Get tuning curve.
% Input - Learning_stage: 'early' or 'late'.

for seed_num = 1:6
    clearvars -except learning_stage seed_num tuning_task2
    
    load('agent_behavior_task2.mat')
    load('agent_activity_task2.mat')
    
    if contains(learning_stage,'early')
        action = behavior_task2{seed_num}.early.action;
        q_first_layer_action_based = activity_task2{seed_num}.early.q_first_layer_action_based;
        q_second_layer_action_based = activity_task2{seed_num}.early.q_second_layer_action_based;
        q_third_layer_action_based = activity_task2{seed_num}.early.q_third_layer_action_based;
    elseif contains(learning_stage,'late')
        action = behavior_task2{seed_num}.late.action;
        q_first_layer_action_based = activity_task2{seed_num}.late.q_first_layer_action_based;
        q_second_layer_action_based = activity_task2{seed_num}.late.q_second_layer_action_based;
        q_third_layer_action_based = activity_task2{seed_num}.late.q_third_layer_action_based;
    end
    
    clear behavior_task2 activity_task2 % Save memory.
    
    % Lick tuning.
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
    
    for layer_num = 1:3
        q_lick{layer_num}(:,1) = nanmean(q_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) <= 0.08)),2);
        q_lick{layer_num}(:,2) = nanmean(q_activity_action_based_concat{layer_num}(:,find(action_concat(:,3) > 0.08)),2);
        % Obtain active cell index.
        q_active_cell{layer_num} = find(sum(q_activity_action_based_concat{layer_num},2) ~= 0);
    end
    
    tuning_task2{seed_num}.q_lick = q_lick;
    tuning_task2{seed_num}.q_active_cell = q_active_cell;
end

end