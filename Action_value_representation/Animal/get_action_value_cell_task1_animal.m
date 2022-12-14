function get_action_value_cell_task1_animal(learning_stage)

close all
clearvars -except learning_stage
clc

% Get action value distribution for each neuron based on its activity in space.
% Input - Learning_stage: 'naive' or 'expert'.
% Load 'tuning_task1' after running 'get_tuning_curve_task1_animal.m'.

load('animal_behavior_task1.mat')
load('animal_activity_task1.mat')
if contains(learning_stage,'naive')
    behavior_task1_temp = behavior_task1.naive;
    activity_task1_temp = activity_task1.naive;
elseif contains(learning_stage,'expert')
    behavior_task1_temp = behavior_task1.expert;
    activity_task1_temp = activity_task1.expert;
end
clear behavior_task1 activity_task1
behavior_task1 = behavior_task1_temp;
activity_task1 = activity_task1_temp;

for animal_num = 1:numel(behavior_task1)
    clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 animal_num action_value_cell_task1
    
    for session_num = 1:numel(behavior_task1{animal_num})
        clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 animal_num session_num action_value_cell_task1
        
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
        
        % Rotate and filter.
        value_function = imrotate(squeeze(nanmean(mean_step_size_from_state)),90);
        image_filter = fspecial('gaussian',1,1);
        filtered_value_function = nanconv(value_function,image_filter,'edge','nanout');
        
        % Downsample filtered_value_function.
        flipped_filtered_value_function = flipud(filtered_value_function); % Adjust for policy.
        downsamp_flipped_filtered_value_function_temp1 = cat(3,flipped_filtered_value_function(1:2:end,1:2:end),flipped_filtered_value_function(2:2:end,2:2:end));
        downsamp_flipped_filtered_value_function_temp2 = cat(3,flipped_filtered_value_function(1:2:end,2:2:end),flipped_filtered_value_function(2:2:end,1:2:end));
        downsamp_flipped_filtered_value_function = nanmean(cat(3,downsamp_flipped_filtered_value_function_temp1,downsamp_flipped_filtered_value_function_temp2),3); % Downsampling by averaging.
        downsamp_flipped_filtered_value_function(4:7,4:7) = 1; % 1 for the reward zone.
        
        % Calculate action values.
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
        
        clearvars -except learning_stage behavior_task1 activity_task1 tuning_task1 animal_num session_num filtered_value_function action_value_function action_value_cell_task1
        
        GLM = activity_task1{animal_num}{session_num};
        xy_object_pos_cell_idx = tuning_task1{animal_num}{session_num}.xy_object_pos_cell_idx;
        object_vel_cell_idx = tuning_task1{animal_num}{session_num}.object_vel_cell_idx;
        tuning_xy_object_pos = tuning_task1{animal_num}{session_num}.tuning_xy_object_pos;
        tuning_object_dir = tuning_task1{animal_num}{session_num}.tuning_object_dir;
        
        % Get conjunctive cells.
        if ~isempty(GLM.activity_matrix{1}) == 1 && ~isempty(GLM.activity_matrix{2}) == 1
            region_num_temp = 1; region = 2;
        elseif ~isempty(GLM.activity_matrix{1}) == 0 && ~isempty(GLM.activity_matrix{2}) == 1
            region_num_temp = 2; region = 2;
        elseif ~isempty(GLM.activity_matrix{1}) == 1 && ~isempty(GLM.activity_matrix{2}) == 0
            region_num_temp = 1; region = 2;
        end
        for region_num = region_num_temp:region
            [~,object_vel_and_xy_object_pos_idx{region_num},xy_object_pos_and_object_vel_idx{region_num}] = intersect(object_vel_cell_idx{region_num},xy_object_pos_cell_idx{region_num});
        end
        
        % Determine action value distribution in active states for each neuron.
        if ~isempty(xy_object_pos_and_object_vel_idx)
            for region_num = region_num_temp:region
                if ~isempty(xy_object_pos_and_object_vel_idx{region_num})
                    for cell_num = 1:length(xy_object_pos_and_object_vel_idx{region_num}) % Conjunctive coding cells.
                        tuning_xy_object_pos_conj_cells_temp_object_vel{region_num}{cell_num} = squeeze(tuning_xy_object_pos{region_num}(xy_object_pos_cell_idx{region_num}(xy_object_pos_and_object_vel_idx{region_num}(cell_num)),:,:));
                        tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} = imrotate(tuning_xy_object_pos_conj_cells_temp_object_vel{region_num}{cell_num},90);
                        flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} = flipud(tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}); % Adjust for action-value.
                        downsamp_flip_tuning_xy_object_pos_object_vel_temp1{region_num}{cell_num} = cat(3,flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}(1:2:end,1:2:end),flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}(2:2:end,2:2:end));
                        downsamp_flip_tuning_xy_object_pos_object_vel_temp2{region_num}{cell_num} = cat(3,flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}(1:2:end,2:2:end),flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}(2:2:end,1:2:end));
                        downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} = nanmean(cat(3,downsamp_flip_tuning_xy_object_pos_object_vel_temp1{region_num}{cell_num},downsamp_flip_tuning_xy_object_pos_object_vel_temp2{region_num}{cell_num}),3); % Downsampling by averaging.
                        norm_downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} = (downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} - min(min(downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num})))./(max(max(downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} - min(min(downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num})))));
                        [map_value_object_vel{region_num}{cell_num},map_index_object_vel{region_num}{cell_num}] = sort(norm_downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num}(:),'descend','MissingPlacement','last');
                        
                        action_dist_active_bins_all_object_vel{region_num}{cell_num} = [];
                        if sum(map_value_object_vel{region_num}{cell_num}(1:5)) >= 4.9999999999 % If map values are all 1.
                            action_dist_active_bins_all_object_vel{region_num}{cell_num} = [];
                        else
                            rows_map_value_object_vel_temp_all = [];
                            cols_map_value_object_vel_temp_all = [];
                            for bin_num = 1:5 % Top 5%.
                                [rows_map_value_object_vel_temp{region_num}{cell_num}{bin_num},cols_map_value_object_vel_temp{region_num}{cell_num}{bin_num}] = find(norm_downsamp_flip_tuning_xy_object_pos_conj_cells_object_vel{region_num}{cell_num} == map_value_object_vel{region_num}{cell_num}(bin_num));
                                rows_map_value_object_vel_temp_all = [rows_map_value_object_vel_temp_all;rows_map_value_object_vel_temp{region_num}{cell_num}{bin_num}];
                                cols_map_value_object_vel_temp_all = [cols_map_value_object_vel_temp_all;cols_map_value_object_vel_temp{region_num}{cell_num}{bin_num}];
                            end
                            rows_map_value_object_vel{region_num}{cell_num} = rows_map_value_object_vel_temp_all(1:5);
                            cols_map_value_object_vel{region_num}{cell_num} = cols_map_value_object_vel_temp_all(1:5);
                            for bin_num = 1:5 % Top 5%.
                                action_dist_active_bins_object_vel{region_num}{cell_num}{bin_num} = action_value_function{cols_map_value_object_vel{region_num}{cell_num}(bin_num)}{rows_map_value_object_vel{region_num}{cell_num}(bin_num)}(1:9);
                                action_dist_active_bins_object_vel{region_num}{cell_num}{bin_num} = action_dist_active_bins_object_vel{region_num}{cell_num}{bin_num}.*map_value_object_vel{region_num}{cell_num}(bin_num); % Weight.
                                action_dist_active_bins_all_object_vel{region_num}{cell_num} = [action_dist_active_bins_all_object_vel{region_num}{cell_num};action_dist_active_bins_object_vel{region_num}{cell_num}{bin_num}];
                            end
                        end
                        
                        mean_action_dist_active_bins_all_object_vel{region_num}{cell_num} = nanmean(action_dist_active_bins_all_object_vel{region_num}{cell_num});
                    end
                else
                    mean_action_dist_active_bins_all_object_vel{region_num} = [];
                end
            end
        else
            mean_action_dist_active_bins_all_object_vel = [];
        end
        
        % Metric.
        for region_num = region_num_temp:region
            % Action value.
            if ~isempty(mean_action_dist_active_bins_all_object_vel{region_num}) == 1
                for cell_num = 1:length(xy_object_pos_and_object_vel_idx{region_num}) % Conjunctive coding cells.
                    if isempty(mean_action_dist_active_bins_all_object_vel{region_num}{cell_num}) == 1 | isnan(mean_action_dist_active_bins_all_object_vel{region_num}{cell_num}) == 1
                        dot_product_object_vel{region_num}(cell_num) = nan;
                        correlation_object_vel{region_num}(cell_num) = nan;
                    else
                        dot_product_object_vel{region_num}(cell_num) = nansum(tuning_object_dir{region_num}(object_vel_cell_idx{region_num}(object_vel_and_xy_object_pos_idx{region_num}(cell_num)),:).*mean_action_dist_active_bins_all_object_vel{region_num}{cell_num});
                        correlation_object_vel{region_num}(cell_num) = corr(tuning_object_dir{region_num}(object_vel_cell_idx{region_num}(object_vel_and_xy_object_pos_idx{region_num}(cell_num)),:)',mean_action_dist_active_bins_all_object_vel{region_num}{cell_num}','row','complete');
                    end
                end
            else
                dot_product_object_vel{region_num} = [];
                correlation_object_vel{region_num} = [];
            end
            
            % State value.
            if ~isempty(xy_object_pos_and_object_vel_idx{region_num}) == 1
                for cell_num = 1:length(xy_object_pos_and_object_vel_idx{region_num}) % Conjunctive coding cells.
                    tuning_xy_object_pos_xy_object_pos_cell{region_num}{cell_num} = imrotate(squeeze(tuning_xy_object_pos{region_num}(xy_object_pos_cell_idx{region_num}(xy_object_pos_and_object_vel_idx{region_num}(cell_num)),:,:)),90);
                    correlation_state_value_space_tuning{region_num}(cell_num) = corr(filtered_value_function(:),tuning_xy_object_pos_xy_object_pos_cell{region_num}{cell_num}(:),'row','complete');
                end
            else
                correlation_state_value_space_tuning{region_num} = [];
            end
        end
        
        action_value_cell_task1{animal_num}{session_num}.action_value_function = action_value_function;
        action_value_cell_task1{animal_num}{session_num}.action_value_dist = mean_action_dist_active_bins_all_object_vel;
        action_value_cell_task1{animal_num}{session_num}.dot_product_object_vel = dot_product_object_vel;
        action_value_cell_task1{animal_num}{session_num}.correlation_object_vel = correlation_object_vel;
        action_value_cell_task1{animal_num}{session_num}.correlation_state_value_space_tuning = correlation_state_value_space_tuning;
        action_value_cell_task1{animal_num}{session_num}.filtered_value_function = filtered_value_function;
    end
end

end