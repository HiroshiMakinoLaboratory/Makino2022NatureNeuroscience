function plot_state_value_and_policy_task1_agent(learning_stage)

close all
clearvars -except learning_stage
clc

% Plot state value and policy.
% Input - Learning_stage: 'naive' or 'expert'.

load('agent_behavior_task1.mat')

% Initialize.
value_function_agent = [];
concat_sum_speed_vec_each_pos_x_agent = [];
concat_sum_speed_vec_each_pos_y_agent = [];

for seed_num = 1:numel(behavior_task1)
    clearvars -except learning_stage behavior_task1 value_function_agent concat_sum_speed_vec_each_pos_x_agent concat_sum_speed_vec_each_pos_y_agent seed_num
    
    if contains(learning_stage,'naive')
        observation = behavior_task1{seed_num}.naive.observation;
        action = behavior_task1{seed_num}.naive.action;
        reward = behavior_task1{seed_num}.naive.reward;
    elseif contains(learning_stage,'expert')
        observation = behavior_task1{seed_num}.expert.observation;
        action = behavior_task1{seed_num}.expert.action;
        reward = behavior_task1{seed_num}.expert.reward;
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
    
    % Plot state value functions.
    value_function_temp = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
    value_function_temp(7:14,7:14) = nan;
    image_filter = fspecial('gaussian',2,2);
    value_function = nanconv(value_function_temp,image_filter,'edge','nanout');
    value_function(7:14,7:14) = 1;
    
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
rectangle('Position',[6.5,6.5,8,8],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor',[0.5,0.5,0.5])
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

% Downsample.
mean_value_function_agent = reshape(nanmean(value_function_agent,2),[20,20]);
for bin = 1:10
    downsample_temp1(:,bin) = nansum(mean_value_function_agent(:,1 + (bin - 1)*2:2 + (bin - 1)*2),2);
end
for bin = 1:10
    downsample_temp2(bin,:) = nansum(downsample_temp1(1 + (bin - 1)*2:2 + (bin - 1)*2,:));
end
downsampled_mean_value_function_agent_temp = downsample_temp2./4;

% Plot mean state values, downsampled.
figure('Position',[500,500,250,250],'Color','w');
image_filter = fspecial('gaussian',1,1);
downsampled_mean_value_function_agent_temp(4:7,4:7) = nan;
downsampled_mean_value_function_agent = nanconv(downsampled_mean_value_function_agent_temp,image_filter,'edge','nanout');
downsampled_mean_value_function_agent(4:7,4:7) = 0.99;
imagesc(downsampled_mean_value_function_agent,[0.1,0.8]);
rectangle('Position',[3.5,3.5,4,4],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor',[0.5,0.5,0.5])
xlabel('x (cm)');
ylabel('y (cm)');
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

% Plot mean policy.
norm_concat_sum_speed_vec_each_pos_x = nanmean(concat_sum_speed_vec_each_pos_x_agent,2)./((nanmean(concat_sum_speed_vec_each_pos_x_agent,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_agent,2).^2).^0.5);
norm_concat_sum_speed_vec_each_pos_y = nanmean(concat_sum_speed_vec_each_pos_y_agent,2)./((nanmean(concat_sum_speed_vec_each_pos_x_agent,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_agent,2).^2).^0.5);

figure('Position',[500,500,250,250],'Color','w');
hq = quiver(concat_sum_speed_vec_each_pos(:,1),concat_sum_speed_vec_each_pos(:,2),norm_concat_sum_speed_vec_each_pos_x,norm_concat_sum_speed_vec_each_pos_y);
hq.LineWidth = 1;
hq.Color = 'k';
hq.MaxHeadSize = 2;
hq.AutoScaleFactor = 0.4;
rectangle('Position',[3.5,3.5,4,4],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor',[0.5,0.5,0.5])
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