function get_action_value_cell_subtask_composite_task_animal

close all
clear all
clc

% Get subtask action value distribution for each neuron based on its activity in space.
% Load 'action_value_cell_task1' after running 'get_action_value_cell_task1_animal.m' for the 'expert' stage.
% Load 'action_value_cell_task2' after running 'get_action_value_cell_task2_animal.m'.
% Load 'tuning_composite_task' after running 'get_tuning_curve_composite_task_animal.m' for the 'early' stage.

load('animal_activity_composite_task.mat')
activity_composite_task_temp = activity_composite_task.pretraining;
tuning_composite_task_temp = tuning_composite_task.pretraining;
clear activity_composite_task tuning_composite_task
activity_composite_task = activity_composite_task_temp;
tuning_composite_task = tuning_composite_task_temp;

for animal_num = 1:numel(activity_composite_task)
    clearvars -except action_value_cell_task1 action_value_cell_task2 activity_composite_task tuning_composite_task animal_num action_value_cell_subtask_composite_task
    
    % Get action-value functions from subtasks.
    % Action values for object movement.
    for action_idx = 1:9
        action_value_function_session = [];
        for session_num = 1:length(action_value_cell_task1{animal_num})
            action_value_function_concat = [];
            for x_bin_num = 1:10
                for y_bin_num = 1:10
                    action_value_function_concat = [action_value_function_concat;action_value_cell_task1{animal_num}{session_num}.action_value_function{x_bin_num}{y_bin_num}(action_idx)];
                end
            end
            action_value_function_session = [action_value_function_session,action_value_function_concat];
        end
        mean_action_value_function(:,action_idx) = mean(action_value_function_session,2);
    end
    clear action_value_function_session action_value_function_concat
    
    % Action values for lick.
    for action_idx = 10:11
        action_value_function_session = [];
        for session_num = 4:5 % Last 2 days.
            action_value_function_concat = [];
            for x_bin_num = 1:10
                for y_bin_num = 1:10
                    action_value_function_concat = [action_value_function_concat;action_value_cell_task2{animal_num}{session_num}.action_value_function{x_bin_num}{y_bin_num}(action_idx - 9)];
                end
            end
            action_value_function_session = [action_value_function_session,action_value_function_concat];
        end
        mean_action_value_function(:,action_idx) = mean(action_value_function_session,2);
    end
    clear action_value_function_session action_value_function_concat
    
    for action_idx = 1:11
        action_value_function(:,:,action_idx) = reshape(mean_action_value_function(:,action_idx),[10,10]);
    end
    
    % Get mean state-value function of task1.
    state_value_function_all = [];
    for session_num = 1:length(action_value_cell_task1{animal_num})
        state_value_function_all = cat(3,state_value_function_all,action_value_cell_task1{animal_num}{session_num}.filtered_value_function);
    end
    filtered_value_function_task1 = nanmean(state_value_function_all,3);
    
    % Get mean state-value function of task2.
    state_value_function_all = [];
    for date_num_state_value = 4:5 % Last 2 days.
        state_value_function_all = cat(3,state_value_function_all,action_value_cell_task2{animal_num}{session_num}.filtered_value_function);
    end
    filtered_value_function_task2 = nanmean(state_value_function_all,3);
    % Convert nan to 0.
    filtered_value_function_task2(isnan(filtered_value_function_task2)) = 0;
    
    % Normalize.
    norm_filtered_value_function_task1 = filtered_value_function_task1./nanmax(filtered_value_function_task1(:));
    norm_filtered_value_function_task2 = filtered_value_function_task2./nanmax(filtered_value_function_task2(:));
    
    % Combine the two state-value functions.
    filtered_value_function = (norm_filtered_value_function_task1 + norm_filtered_value_function_task2)./2;
    
    early_session = [2,1,1,1,1,2,2];
    session_num = early_session(animal_num);
    
    clearvars -except action_value_cell_task1 action_value_cell_task2 activity_composite_task tuning_composite_task animal_num action_value_function session_num filtered_value_function action_value_cell_subtask_composite_task
    
    GLM = activity_composite_task{animal_num}{session_num};
    xy_object_pos_cell_idx = tuning_composite_task{animal_num}.xy_object_pos_cell_idx;
    object_vel_cell_idx = tuning_composite_task{animal_num}.object_vel_cell_idx;
    lick_onset_cell_idx = tuning_composite_task{animal_num}.lick_onset_cell_idx;
    tuning_xy_object_pos = tuning_composite_task{animal_num}.tuning_xy_object_pos;
    tuning_object_dir = tuning_composite_task{animal_num}.tuning_object_dir;
    tuning_lick_onset = tuning_composite_task{animal_num}.tuning_lick_onset;
    
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
        [~,lick_onset_and_xy_object_pos_idx{region_num},xy_object_pos_and_lick_onset_idx{region_num}] = intersect(lick_onset_cell_idx{region_num},xy_object_pos_cell_idx{region_num});
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
                            action_dist_active_bins_object_vel{region_num}{cell_num}{bin_num} = squeeze(action_value_function(rows_map_value_object_vel{region_num}{cell_num}(bin_num),cols_map_value_object_vel{region_num}{cell_num}(bin_num),1:9))';
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
    
    if ~isempty(xy_object_pos_and_lick_onset_idx)
        for region_num = region_num_temp:region
            if ~isempty(xy_object_pos_and_lick_onset_idx{region_num})
                for cell_num = 1:length(xy_object_pos_and_lick_onset_idx{region_num}) % Conjunctive coding cells.
                    tuning_xy_object_pos_conj_cells_temp_lick_onset{region_num}{cell_num} = squeeze(tuning_xy_object_pos{region_num}(xy_object_pos_cell_idx{region_num}(xy_object_pos_and_lick_onset_idx{region_num}(cell_num)),:,:));
                    tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} = imrotate(tuning_xy_object_pos_conj_cells_temp_lick_onset{region_num}{cell_num},90);
                    flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} = flipud(tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}); % Adjust for action-value.
                    downsamp_flip_tuning_xy_object_pos_lick_onset_temp1{region_num}{cell_num} = cat(3,flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}(1:2:end,1:2:end),flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}(2:2:end,2:2:end));
                    downsamp_flip_tuning_xy_object_pos_lick_onset_temp2{region_num}{cell_num} = cat(3,flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}(1:2:end,2:2:end),flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}(2:2:end,1:2:end));
                    downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} = nanmean(cat(3,downsamp_flip_tuning_xy_object_pos_lick_onset_temp1{region_num}{cell_num},downsamp_flip_tuning_xy_object_pos_lick_onset_temp2{region_num}{cell_num}),3); % Downsampling by averaging.
                    norm_downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} = (downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} - min(min(downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num})))./(max(max(downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} - min(min(downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num})))));
                    [map_value_lick_onset{region_num}{cell_num},map_index_lick_onset{region_num}{cell_num}] = sort(norm_downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num}(:),'descend','MissingPlacement','last');
                    
                    action_dist_active_bins_all_lick_onset{region_num}{cell_num} = [];
                    if sum(map_value_lick_onset{region_num}{cell_num}(1:5)) >= 4.9999999999 % If map values are all 1.
                        action_dist_active_bins_all_lick_onset{region_num}{cell_num} = [];
                    else
                        rows_map_value_lick_onset_temp_all = [];
                        cols_map_value_lick_onset_temp_all = [];
                        for bin_num = 1:5 % Top 5%.
                            [rows_map_value_lick_onset_temp{region_num}{cell_num}{bin_num},cols_map_value_lick_onset_temp{region_num}{cell_num}{bin_num}] = find(norm_downsamp_flip_tuning_xy_object_pos_conj_cells_lick_onset{region_num}{cell_num} == map_value_lick_onset{region_num}{cell_num}(bin_num));
                            rows_map_value_lick_onset_temp_all = [rows_map_value_lick_onset_temp_all;rows_map_value_lick_onset_temp{region_num}{cell_num}{bin_num}];
                            cols_map_value_lick_onset_temp_all = [cols_map_value_lick_onset_temp_all;cols_map_value_lick_onset_temp{region_num}{cell_num}{bin_num}];
                        end
                        rows_map_value_lick_onset{region_num}{cell_num} = rows_map_value_lick_onset_temp_all(1:5);
                        cols_map_value_lick_onset{region_num}{cell_num} = cols_map_value_lick_onset_temp_all(1:5);
                        for bin_num = 1:5 % Top 5%.
                            action_dist_active_bins_lick_onset{region_num}{cell_num}{bin_num} = squeeze(action_value_function(rows_map_value_lick_onset{region_num}{cell_num}(bin_num),cols_map_value_lick_onset{region_num}{cell_num}(bin_num),10:11))';
                            action_dist_active_bins_lick_onset{region_num}{cell_num}{bin_num} = action_dist_active_bins_lick_onset{region_num}{cell_num}{bin_num}.*map_value_lick_onset{region_num}{cell_num}(bin_num); % Weight.
                            action_dist_active_bins_all_lick_onset{region_num}{cell_num} = [action_dist_active_bins_all_lick_onset{region_num}{cell_num};action_dist_active_bins_lick_onset{region_num}{cell_num}{bin_num}];
                        end
                    end
                    
                    mean_action_dist_active_bins_all_lick_onset{region_num}{cell_num} = nanmean(action_dist_active_bins_all_lick_onset{region_num}{cell_num});
                end
            else
                mean_action_dist_active_bins_all_lick_onset{region_num} = [];
            end
        end
    else
        mean_action_dist_active_bins_all_lick_onset = [];
    end
    
    % Metric.
    for region_num = region_num_temp:region
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
        
        if ~isempty(mean_action_dist_active_bins_all_lick_onset{region_num}) == 1
            for cell_num = 1:length(xy_object_pos_and_lick_onset_idx{region_num}) % Conjunctive coding cells.
                if isempty(mean_action_dist_active_bins_all_lick_onset{region_num}{cell_num}) == 1 | isnan(mean_action_dist_active_bins_all_lick_onset{region_num}{cell_num}) == 1
                    dot_product_lick_onset{region_num}(cell_num) = nan;
                    correlation_lick_onset{region_num}(cell_num) = nan;
                else
                    dot_product_lick_onset{region_num}(cell_num) = nansum(tuning_lick_onset{region_num}(lick_onset_cell_idx{region_num}(lick_onset_and_xy_object_pos_idx{region_num}(cell_num)),:).*mean_action_dist_active_bins_all_lick_onset{region_num}{cell_num});
                    correlation_lick_onset{region_num}(cell_num) = corr(tuning_lick_onset{region_num}(lick_onset_cell_idx{region_num}(lick_onset_and_xy_object_pos_idx{region_num}(cell_num)),:)',mean_action_dist_active_bins_all_lick_onset{region_num}{cell_num}','row','complete');
                end
            end
        else
            dot_product_lick_onset{region_num} = [];
            correlation_lick_onset{region_num} = [];
        end
        
        % State value.
        if ~isempty(xy_object_pos_and_object_vel_idx{region_num}) == 1
            for cell_num = 1:length(xy_object_pos_and_object_vel_idx{region_num}) % Conjunctive coding cells.
                tuning_xy_object_pos_xy_object_pos_cell_object_vel{region_num}{cell_num} = imrotate(squeeze(tuning_xy_object_pos{region_num}(xy_object_pos_cell_idx{region_num}(xy_object_pos_and_object_vel_idx{region_num}(cell_num)),:,:)),90);
                correlation_state_value_space_tuning_object_vel{region_num}(cell_num) = corr(filtered_value_function(:),tuning_xy_object_pos_xy_object_pos_cell_object_vel{region_num}{cell_num}(:),'row','complete');
            end
        else
            correlation_state_value_space_tuning_object_vel{region_num} = [];
        end
        
        if ~isempty(xy_object_pos_and_lick_onset_idx{region_num}) == 1
            for cell_num = 1:length(xy_object_pos_and_lick_onset_idx{region_num}) % Conjunctive coding cells.
                tuning_xy_object_pos_xy_object_pos_cell_lick_onset{region_num}{cell_num} = imrotate(squeeze(tuning_xy_object_pos{region_num}(xy_object_pos_cell_idx{region_num}(xy_object_pos_and_lick_onset_idx{region_num}(cell_num)),:,:)),90);
                correlation_state_value_space_tuning_lick_onset{region_num}(cell_num) = corr(filtered_value_function(:),tuning_xy_object_pos_xy_object_pos_cell_lick_onset{region_num}{cell_num}(:),'row','complete');
            end
        else
            correlation_state_value_space_tuning_lick_onset{region_num} = [];
        end
    end
    
    action_value_cell_subtask_composite_task{animal_num}.action_value_function = action_value_function;
    action_value_cell_subtask_composite_task{animal_num}.action_value_dist_object_vel = mean_action_dist_active_bins_all_object_vel;
    action_value_cell_subtask_composite_task{animal_num}.action_value_dist_lick = mean_action_dist_active_bins_all_lick_onset;
    action_value_cell_subtask_composite_task{animal_num}.dot_product_object_vel = dot_product_object_vel;
    action_value_cell_subtask_composite_task{animal_num}.correlation_object_vel = correlation_object_vel;
    action_value_cell_subtask_composite_task{animal_num}.dot_product_lick_onset = dot_product_lick_onset;
    action_value_cell_subtask_composite_task{animal_num}.correlation_lick_onset = correlation_lick_onset;
    action_value_cell_subtask_composite_task{animal_num}.correlation_state_value_space_tuning_object_vel = correlation_state_value_space_tuning_object_vel;
    action_value_cell_subtask_composite_task{animal_num}.correlation_state_value_space_tuning_lick_onset = correlation_state_value_space_tuning_lick_onset;
end

end