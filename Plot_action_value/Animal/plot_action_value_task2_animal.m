function plot_action_value_task2_animal

close all
clear all
clc

% Plot action value.

load('animal_behavior_task2.mat')

% Initialize.
action_value_function_animal = [];

for animal_num = 1:numel(behavior_task2)
    clearvars -except behavior_task2 action_value_function_animal animal_num
    
    % Initialize.
    action_value_function = [];
    
    session_num = 5;
    clearvars -except behavior_task2  ...
        action_value_function_animal animal_num ...
        action_value_function session_num
    
    % Determine correct and incorrect trials.
    correct_trial_temp = zeros(1,behavior_task2{animal_num}{session_num}.bpod.nTrials);
    for trial_num = 1:behavior_task2{animal_num}{session_num}.bpod.nTrials
        correct_trial_temp(trial_num) = ~isnan(behavior_task2{animal_num}{session_num}.bpod.RawEvents.Trial{trial_num}.States.Reward(1));
    end
    all_trial = [1:behavior_task2{animal_num}{session_num}.bpod.nTrials];
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
    trial = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,trial_ch);
    x_stage = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,x_stage_ch);
    y_stage = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,y_stage_ch);
    x_joystick = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,x_joystick_ch);
    y_joystick = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,y_joystick_ch);
    lick = behavior_task2{animal_num}{session_num}.wavesurfer.sweep_0001.analogScans(:,lick_ch);
    
    % Sampling frequency of WaveSurfer data.
    fs_behavior = behavior_task2{animal_num}{session_num}.wavesurfer.header.AcquisitionSampleRate;
    
    % Determine trial begining and end.
    thresh = 2.5;
    trial_str = trial > thresh; % Binarize.
    trial_begin = strfind(trial_str',[0,1]) + 1;
    trial_end = strfind(trial_str',[1,0]);
    
    % Analyze object trajectory.
    x_stage_smooth = smooth(double(x_stage),fs_behavior*0.01); % Moving average across 10 ms.
    y_stage_smooth = smooth(double(y_stage),fs_behavior*0.01); % Moving average across 10 ms.
    
    % Calculate state values.
    for trial_num = 1:behavior_task2{animal_num}{session_num}.bpod.nTrials
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
    for trial_num = 1:behavior_task2{animal_num}{session_num}.bpod.nTrials
        for x_bin_num = 1:20
            for y_bin_num = 1:20
                mean_step_size_from_state(trial_num,x_bin_num,y_bin_num) = nan;
            end
        end
        mean_step_size_from_state(trial_num,10,1) = gamma.^(length(x_bin{trial_num}) - 2*100); % Subtract 2 seconds (100 steps = 10 ms for 1 second).
        mean_step_size_from_state(trial_num,11,1) = gamma.^(length(x_bin{trial_num}) - 2*100); % Subtract 2 seconds (100 steps = 10 ms for 1 second).
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
    
    % Calculate action values.
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            action_value_function{x_bin_num}{y_bin_num}(1,1:2) = nan; % 1 = no lick, 2 = lick.
            if x_bin_num == 5 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num); % Same state.
                action_value_function{x_bin_num}{y_bin_num}(2) = 1;
            elseif x_bin_num == 6 && y_bin_num == 1
                action_value_function{x_bin_num}{y_bin_num}(1) = 0 + gamma.*downsamp_flipped_filtered_value_function(y_bin_num,x_bin_num - 1); % Same state.
                action_value_function{x_bin_num}{y_bin_num}(2) = 1;
            end
        end
    end
    
    action_value_function_concat = [];
    for x_bin_num = 1:10
        for y_bin_num = 1:10
            action_value_function_concat = [action_value_function_concat;action_value_function{x_bin_num}{y_bin_num}];
        end
    end
    
    % Concatenate across animals.
    action_value_function_animal = cat(3,action_value_function_animal,action_value_function_concat);
end

% Mean across animals.
mean_action_value_function_animal = nanmean(action_value_function_animal,3);

image_filter = fspecial('gaussian',1,1);

% Plot mean action values averaged across actions.
mean_mean_action_value_function_animal = nanmean(mean_action_value_function_animal,2);
figure('Position',[500,500,250,250],'Color','w');
reshaped_mean_mean_action_value_function_animal = flipud(reshape(mean_mean_action_value_function_animal,[10,10]));
filt_reshaped_mean_mean_action_value_function_animal = nanconv(reshaped_mean_mean_action_value_function_animal,image_filter,'edge','nanout');
imagesc(filt_reshaped_mean_mean_action_value_function_animal,[0.4665,1])
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
for action_idx = 1:2
    figure('Position',[500,500,250,250],'Color','w');
    reshaped_mean_action_function_animal_mean_subtract = flipud(reshape((mean_action_value_function_animal(:,action_idx) - mean_mean_action_value_function_animal),[10,10]));
    filt_reshaped_mean_action_function_animal_mean_subtract = nanconv(reshaped_mean_action_function_animal_mean_subtract,image_filter,'edge','nanout');
    imagesc(filt_reshaped_mean_action_function_animal_mean_subtract,[-0.2668,0.2668])
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