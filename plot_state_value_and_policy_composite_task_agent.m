function plot_state_value_and_policy_composite_task_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Plot state value and policy.
% Input - Learning_stage: 'early' or 'late'.

cd('/Volumes/G-DRIVE USB-C/Makino_nature_neuroscience_data/Submission')
load('agent_behavior_composite_task.mat')

% Initialize.
value_function_agent = [];
concat_sum_speed_vec_each_pos_x_agent = [];
concat_sum_speed_vec_each_pos_y_agent = [];

for seed_num = 1:numel(behavior_composite_task.pretraining)
    clearvars -except learning_stage behavior_composite_task value_function_agent concat_sum_speed_vec_each_pos_x_agent concat_sum_speed_vec_each_pos_y_agent seed_num
    
    if contains(learning_stage,'early')
        observation = behavior_composite_task.pretraining{seed_num}.early.observation;
        action = behavior_composite_task.pretraining{seed_num}.early.action;
        reward = behavior_composite_task.pretraining{seed_num}.early.reward;
    elseif contains(learning_stage,'late')
        observation = behavior_composite_task.pretraining{seed_num}.late.observation;
        action = behavior_composite_task.pretraining{seed_num}.late.action;
        reward = behavior_composite_task.pretraining{seed_num}.late.reward;
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
    image_filter = fspecial('gaussian',2,2);
    value_function = nanconv(value_function_temp,image_filter,'edge','nanout');
    
    state_action = [];
    for trial_num = 1:length(observation)
        % Get binned position.
        [~,~,~,x_bin_vector_field{trial_num},y_bin_vector_field{trial_num}] = histcounts2(x_trial{trial_num},y_trial{trial_num},'XBinEdges',[-1:0.2:1],'YBinEdges',[-1:0.2:1]);
        x_bin_vector_field{trial_num} = x_bin_vector_field{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
        y_bin_vector_field{trial_num} = y_bin_vector_field{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
        
        state_action = [state_action;x_bin_vector_field{trial_num},y_bin_vector_field{trial_num},action{trial_num}];
    end
    
    concat_sum_speed_vec_each_pos = [];
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            sum_speed_vec_each_pos{x_bin_num}{y_bin_num} = [x_bin_num,y_bin_num,nansum(state_action(find(state_action(:,1) == x_bin_num & state_action(:,2) == y_bin_num),3:4))];
            concat_sum_speed_vec_each_pos = [concat_sum_speed_vec_each_pos;sum_speed_vec_each_pos{x_bin_num}{y_bin_num}];
        end
    end
    
    % Concatenate across agents.
    value_function_agent = [value_function_agent,value_function(:)];
    concat_sum_speed_vec_each_pos_x_agent = [concat_sum_speed_vec_each_pos_x_agent,concat_sum_speed_vec_each_pos(:,3)];
    concat_sum_speed_vec_each_pos_y_agent = [concat_sum_speed_vec_each_pos_y_agent,concat_sum_speed_vec_each_pos(:,4)];
end

% Plot mean state values.
figure('Position',[500,500,250,250],'Color','w');
imagesc(reshape(nanmean(value_function_agent,2),[20,20]),[0.3,1]);
xlabel('x (cm)');
ylabel('y (cm)')
xlim([0.5,20.5]);
ylim([0.5,20.5]);
axis square
ax = gca;
ax.Color = 'w';
ax.FontSize = 14;
ax.LineWidth = 1;
ax.XColor = 'k';
ax.YColor = 'k';
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;
ax.XTick = [0.5,10.5,20.5];
ax.XTickLabel = {'0','5','10'};
ax.YTick = [0.5,10.5,20.5];
ax.YTickLabel = {'10','5','0'};
colormap('redblue')

% Plot mean policy.
norm_concat_sum_speed_vec_each_pos_x = nanmean(concat_sum_speed_vec_each_pos_x_agent,2)./((nanmean(concat_sum_speed_vec_each_pos_x_agent,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_agent,2).^2).^0.5);
norm_concat_sum_speed_vec_each_pos_y = nanmean(concat_sum_speed_vec_each_pos_y_agent,2)./((nanmean(concat_sum_speed_vec_each_pos_x_agent,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_agent,2).^2).^0.5);

figure('Position',[500,500,250,250],'Color','w');
hq = quiver(concat_sum_speed_vec_each_pos(:,1),concat_sum_speed_vec_each_pos(:,2),norm_concat_sum_speed_vec_each_pos_x,norm_concat_sum_speed_vec_each_pos_y);
hq.LineWidth = 1;
hq.Color = 'k';
hq.MaxHeadSize = 2;
hq.AutoScaleFactor = 0.4;
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
ax.YTickLabel = {'0','5','10'};

end