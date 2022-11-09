function plot_state_value_and_policy_task1_animal(learning_stage)

close all
clearvars -except learning_stage
clc

% Plot state value and policy.
% Input - Learning_stage: 'naive' or 'expert'.

load('animal_behavior_task1.mat')

if contains(learning_stage,'naive')
    behavior_task1_temp = behavior_task1.naive;
    clear behavior_task1
    behavior_task1 = behavior_task1_temp;
elseif contains(learning_stage,'expert')
    behavior_task1_temp = behavior_task1.expert;
    clear behavior_task1
    behavior_task1 = behavior_task1_temp;
end

% Initialize.
value_function_animal_session = [];
concat_sum_speed_vec_each_pos_x_animal_session = [];
concat_sum_speed_vec_each_pos_y_animal_session = [];

for animal_num = 1:numel(behavior_task1)
    clearvars -except behavior_task1 value_function_animal_session concat_sum_speed_vec_each_pos_x_animal_session concat_sum_speed_vec_each_pos_y_animal_session animal_num
    
    % Initialize.
    value_function_session = [];
    concat_sum_speed_vec_each_pos_x_session = [];
    concat_sum_speed_vec_each_pos_y_session = [];
    
    for session_num = 1:numel(behavior_task1{animal_num})
        clearvars -except behavior_task1  ...
            value_function_animal_session concat_sum_speed_vec_each_pos_x_animal_session concat_sum_speed_vec_each_pos_y_animal_session animal_num ...
            value_function_session concat_sum_speed_vec_each_pos_x_session concat_sum_speed_vec_each_pos_y_session session_num
        
        % Determine correct and incorrect trials.
        correct_trial_temp = zeros(1,behavior_task1{animal_num}{session_num}.bpod.nTrials);
        for trial_num = 1:behavior_task1{animal_num}{session_num}.bpod.nTrials
            correct_trial_temp(trial_num) = ~isnan(behavior_task1{animal_num}{session_num}.bpod.RawEvents.Trial{trial_num}.States.Reward(1));
        end
        all_trial = [1:behavior_task1{animal_num}{session_num}.bpod.nTrials];
        correct_trial = find(correct_trial_temp);
        incorrect_trial = all_trial(~ismember(all_trial,correct_trial));
        
        % DAQ channels in WaveSurfer.
        trial_ch = 1;
        x_stage_ch = 2;
        y_stage_ch = 3;
        x_joystick_ch = 5;
        y_joystick_ch = 6;
        lick_ch = 7;
        
        % Read from WaveSurfer data.
        trial = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,trial_ch);
        x_stage = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,x_stage_ch);
        y_stage = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,y_stage_ch);
        x_joystick = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,x_joystick_ch);
        y_joystick = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,y_joystick_ch);
        lick = behavior_task1{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,lick_ch);
        
        % Sampling frequency of WaveSurfer data.
        fs_behavior = behavior_task1{animal_num}{session_num}.wavesurfer.header.AcquisitionSampleRate;
        
        % Determine trial begining and end.
        thresh = 2.5;
        trial_str = trial > thresh; % Binarize.
        trial_begin = strfind(trial_str',[0,1]) + 1;
        trial_end = strfind(trial_str',[1,0]);
        
        % Analyze object trajectory.
        x_stage_smooth = smooth(double(x_stage),fs_behavior*0.01); % Moving average across 10 ms.
        y_stage_smooth = smooth(double(y_stage),fs_behavior*0.01); % Moving average across 10 ms.
        
        % Calculate state values.
        for trial_num = 1:behavior_task1{animal_num}{session_num}.bpod.nTrials
            x_stage_trial{trial_num} = x_stage_smooth((trial_begin(trial_num)):trial_end(trial_num));
            y_stage_trial{trial_num} = y_stage_smooth((trial_begin(trial_num)):trial_end(trial_num));
            x_stage_trial_10ms{trial_num} = x_stage_trial{trial_num}(1:fs_behavior*0.01:end); % Sample x stage position every 10 ms.
            y_stage_trial_10ms{trial_num} = y_stage_trial{trial_num}(1:fs_behavior*0.01:end); % Sample y stage position every 10 ms.
            x_stage_trial_10ms_speed{trial_num} = diff(x_stage_trial_10ms{trial_num},1);
            y_stage_trial_10ms_speed{trial_num} = diff(y_stage_trial_10ms{trial_num},1);
            
            % Get object speed vectors for each position.
            [~,~,~,x_bin{trial_num},y_bin{trial_num}] = histcounts2(x_stage_trial_10ms{trial_num},y_stage_trial_10ms{trial_num},'XBinEdges',[0:0.25:5],'YBinEdges',[0:0.25:5]);
            x_bin{trial_num} = x_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
            y_bin{trial_num} = y_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
        end
        
        % Get state-value function.
        gamma = 0.99; % Discount factor.
        for trial_num = 1:behavior_task1{animal_num}{session_num}.bpod.nTrials
            for x_bin_num = 1:20
                for y_bin_num = 1:20
                    mean_step_size_from_state(trial_num,x_bin_num,y_bin_num) = mean(gamma.^(length(x_bin{trial_num}) - find(x_bin{trial_num} == x_bin_num & y_bin{trial_num} == y_bin_num)));
                end
            end
            mean_step_size_from_state(trial_num,7:14,7:14) = 1; % Reward zone.
        end
        
        % Incorporate miss trials.
        if ~isempty(incorrect_trial) == 1
            for incorrect_trial_num = 1:length(incorrect_trial)
                mean_step_size_from_state(incorrect_trial(incorrect_trial_num),:,:) = zeros(1,20,20);
            end
        end
        
        % Rotate.
        value_function = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
        
        % Get object speed vectors over 100 ms.
        clear x_stage_trial y_stage_trial x_bin y_bin
        concat_xy_speed_bin = [];
        for trial_num = 1:behavior_task1{animal_num}{session_num}.bpod.nTrials
            x_stage_trial{trial_num} = x_stage_smooth((trial_begin(trial_num)):trial_end(trial_num));
            y_stage_trial{trial_num} = y_stage_smooth((trial_begin(trial_num)):trial_end(trial_num));
            x_stage_trial_100ms{trial_num} = x_stage_trial{trial_num}(1:fs_behavior*0.1:end); % Sample x stage position every 100 ms.
            y_stage_trial_100ms{trial_num} = y_stage_trial{trial_num}(1:fs_behavior*0.1:end); % Sample y stage position every 100 ms.
            x_stage_trial_100ms_speed{trial_num} = diff(x_stage_trial_100ms{trial_num},1);
            y_stage_trial_100ms_speed{trial_num} = diff(y_stage_trial_100ms{trial_num},1);
            
            % Get object speed vectors for each position.
            [~,~,~,x_bin{trial_num},y_bin{trial_num}] = histcounts2(x_stage_trial_100ms{trial_num},y_stage_trial_100ms{trial_num},'XBinEdges',[0:0.5:5],'YBinEdges',[0:0.5:5]);
            x_bin{trial_num} = x_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
            y_bin{trial_num} = y_bin{trial_num}(1:(end - 1)); % Corresponding the origin of the speed as the speed vector has one fewer time point.
            xy_speed_bin{trial_num} = [x_stage_trial_100ms_speed{trial_num},y_stage_trial_100ms_speed{trial_num},x_bin{trial_num},y_bin{trial_num}]; % Concatenate across movements.
            concat_xy_speed_bin = [concat_xy_speed_bin;xy_speed_bin{trial_num}]; % Concatenate across trials.
        end
        
        % Get a vector field.
        for x_bin_num = 1:10
            for y_bin_num = 1:10
                idx{x_bin_num}{y_bin_num} = nan;
                idx{x_bin_num}{y_bin_num} = find(concat_xy_speed_bin(:,3) == x_bin_num & concat_xy_speed_bin(:,4) == y_bin_num);
            end
        end
        
        % Get summed speed vectors at each position.
        concat_sum_speed_vec_each_pos = [];
        for x_bin_num = 1:10
            for y_bin_num = 1:10
                if isempty(idx{x_bin_num}{y_bin_num}) == 1
                    sum_xy_speed{x_bin_num}{y_bin_num} = [nan,nan];
                elseif size(concat_xy_speed_bin(idx{x_bin_num}{y_bin_num},1)) == 1 % If only one.
                    sum_xy_speed{x_bin_num}{y_bin_num} = concat_xy_speed_bin(idx{x_bin_num}{y_bin_num},1:2);
                else % If more than one.
                    sum_xy_speed{x_bin_num}{y_bin_num} = nansum(concat_xy_speed_bin(idx{x_bin_num}{y_bin_num},1:2));
                end
                sum_speed_vec_each_pos{x_bin_num}{y_bin_num} = [x_bin_num,y_bin_num,sum_xy_speed{x_bin_num}{y_bin_num}];
                concat_sum_speed_vec_each_pos = [concat_sum_speed_vec_each_pos;sum_speed_vec_each_pos{x_bin_num}{y_bin_num}];
            end
        end
        
        % Concatenate across sessions.
        concat_sum_speed_vec_each_pos_x_session = [concat_sum_speed_vec_each_pos_x_session,concat_sum_speed_vec_each_pos(:,3)];
        concat_sum_speed_vec_each_pos_y_session = [concat_sum_speed_vec_each_pos_y_session,concat_sum_speed_vec_each_pos(:,4)];
        value_function_session = [value_function_session,value_function(:)];
    end
    
    % Concatenate across animals.
    concat_sum_speed_vec_each_pos_x_animal_session = [concat_sum_speed_vec_each_pos_x_animal_session,nanmean(concat_sum_speed_vec_each_pos_x_session,2)];
    concat_sum_speed_vec_each_pos_y_animal_session = [concat_sum_speed_vec_each_pos_y_animal_session,nanmean(concat_sum_speed_vec_each_pos_y_session,2)];
    value_function_animal_session = [value_function_animal_session,nanmean(value_function_session,2)];
end

% Plot mean state values.
figure('Position',[500,500,250,250],'Color','w');
value_function = reshape(nanmean(value_function_animal_session,2),[20,20]);
value_function(7:14,7:14) = nan;
image_filter = fspecial('gaussian',2,2);
filtered_value_function = nanconv(value_function,image_filter,'edge','nanout');
filtered_value_function(7:14,7:14) = 1;
imagesc(filtered_value_function,[0,1])
rectangle('Position',[6.5,6.5,8,8],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor',[0.5,0.5,0.5])
xlabel('cm');
ylabel('cm');
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
norm_concat_sum_speed_vec_each_pos_x = nanmean(concat_sum_speed_vec_each_pos_x_animal_session,2)./((nanmean(concat_sum_speed_vec_each_pos_x_animal_session,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_animal_session,2).^2).^0.5);
norm_concat_sum_speed_vec_each_pos_y = nanmean(concat_sum_speed_vec_each_pos_y_animal_session,2)./((nanmean(concat_sum_speed_vec_each_pos_x_animal_session,2).^2 + nanmean(concat_sum_speed_vec_each_pos_y_animal_session,2).^2).^0.5);

figure('Position',[500,500,250,250],'Color','w');
hq = quiver(concat_sum_speed_vec_each_pos(:,1),concat_sum_speed_vec_each_pos(:,2),norm_concat_sum_speed_vec_each_pos_x,norm_concat_sum_speed_vec_each_pos_y);
hq.LineWidth = 1;
hq.Color = 'k';
hq.MaxHeadSize = 2;
hq.AutoScaleFactor = 0.4;
rectangle('Position',[3.5,3.5,4,4],'LineWidth',1,'FaceColor',[0.5,0.5,0.5],'EdgeColor',[0.5,0.5,0.5])
xlabel('cm');
ylabel('cm');
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