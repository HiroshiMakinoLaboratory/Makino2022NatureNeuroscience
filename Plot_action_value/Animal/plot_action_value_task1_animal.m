function plot_action_value_task1_animal

close all
clear all
clc

% Plot action value.

load('animal_behavior_task1.mat')

behavior_task1_temp = behavior_task1.expert;
clear behavior_task1
behavior_task1 = behavior_task1_temp;

% Initialize.
action_value_function_animal_session = [];

for animal_num = 1:numel(behavior_task1)
    clearvars -except behavior_task1 action_value_function_animal_session animal_num
    
    % Initialize.
    action_value_function_session = [];
    
    for session_num = 1:numel(behavior_task1{animal_num})
        clearvars -except behavior_task1  ...
            action_value_function_animal_session animal_num ...
            action_value_function_session session_num
        
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
        image_filter = fspecial('gaussian',2,2);
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
        
        action_value_function_concat = [];
        for x_bin_num = 1:10
            for y_bin_num = 1:10
                action_value_function_concat = [action_value_function_concat;action_value_function{x_bin_num}{y_bin_num}];
            end
        end
        
        % Concatenate across sessions.
        action_value_function_session = cat(3,action_value_function_session,action_value_function_concat);
    end
    
    % Mean across sessions.
    mean_action_value_function_session = nanmean(action_value_function_session,3);
    
    % Concatenate across animals.
    action_value_function_animal_session = cat(3,action_value_function_animal_session,mean_action_value_function_session);
end

% Mean across animals.
mean_action_value_function_animal_session = nanmean(action_value_function_animal_session,3);

image_filter = fspecial('gaussian',1,1);

% Plot mean action values averaged across actions.
mean_mean_action_value_function_animal_session = nanmean(mean_action_value_function_animal_session,2);
figure('Position',[500,500,250,250],'Color','w');
reshaped_mean_mean_action_value_function_animal_session = flipud(reshape(mean_mean_action_value_function_animal_session,[10,10]));
reshaped_mean_mean_action_value_function_animal_session(4:7,4:7) = nan;
filt_reshaped_mean_mean_action_value_function_animal_session = nanconv(reshaped_mean_mean_action_value_function_animal_session,image_filter,'edge','nanout');
filt_reshaped_mean_mean_action_value_function_animal_session(4:7,4:7) = 0.99;
imagesc(filt_reshaped_mean_mean_action_value_function_animal_session,[0,1])
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

% Plot average-subtracted mean action values.
mean_mean_action_value_function_animal_session_movement = mean(mean_action_value_function_animal_session(:,1:8),2);
for action_idx = 1:8
    figure('Position',[500,500,250,250],'Color','w');
    reshaped_mean_action_function_animal_session_mean_subtract = flipud(reshape((mean_action_value_function_animal_session(:,action_idx) - mean_mean_action_value_function_animal_session_movement),[10,10]));
    reshaped_mean_action_function_animal_session_mean_subtract(4:7,4:7) = nan;
    filt_reshaped_mean_action_function_animal_session_mean_subtract = nanconv(reshaped_mean_action_function_animal_session_mean_subtract,image_filter,'edge','nanout');
    filt_reshaped_mean_action_function_animal_session_mean_subtract(4:7,4:7) = 0;
    imagesc(filt_reshaped_mean_action_function_animal_session_mean_subtract,[-0.3,0.3])
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
end

end