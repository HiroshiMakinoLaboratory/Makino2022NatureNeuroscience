function plot_state_value_task2_animal(session_num)

close all
clearvars -except session_num
clc

% Plot state value.
% Input - Session number: 1 or 5.

load('animal_behavior_task2.mat')

% Initialize.
value_function_animal_session = [];

for animal_num = 1:numel(behavior_task2)
    clearvars -except session_num behavior_task2 value_function_animal_session animal_num
    
    % Initialize.
    value_function_session = [];
    
    clearvars -except session_num behavior_task2  ...
        value_function_animal_session animal_num ...
        value_function_session session_num
    
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
    image_filter = fspecial('gaussian',1,1);
    value_function = nanconv(value_function,image_filter,'edge','nanout');
    
    % Concatenate across sessions.
    value_function_session = [value_function_session,value_function(:)];
    
    % Concatenate across animals.
    value_function_animal_session = [value_function_animal_session,nanmean(value_function_session,2)];
end

mean_value_function = reshape(nanmean(value_function_animal_session,2),[20,20]);

% Downsample.
downsamp_mean_state_value_function_all_temp1 = cat(3,mean_value_function(1:2:end,1:2:end),mean_value_function(2:2:end,2:2:end));
downsamp_mean_state_value_function_all_temp2 = cat(3,mean_value_function(1:2:end,2:2:end),mean_value_function(2:2:end,1:2:end));
downsamp_mean_state_value_function_all = nanmean(cat(3,downsamp_mean_state_value_function_all_temp1,downsamp_mean_state_value_function_all_temp2),3); % Downsampling by averaging.

% Plot mean state values, downsampled.
figure('Position',[500,500,250,250],'Color','w');
image_filter = fspecial('gaussian',1,1);
filtered_mean_value_function_downsampled = nanconv(downsamp_mean_state_value_function_all,image_filter,'edge','nanout');
imagesc(filtered_mean_value_function_downsampled,[0.3,0.5])
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
ax.YTickLabel = {'10','5','0'};
colormap('redblue')

end