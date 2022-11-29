function get_direction_tuning_selectivity_task1_animal

close all
clear all
clc

% Get direction tuning selectivity.
% Load 'tuning_task1' after running 'get_tuning_curve_task1_animal.m'.

% Initialize.
selectivity_index_animal_session = [];

for animal_num = 1:numel(tuning_task1)
    clearvars -except policy tuning_task1 selectivity_index_animal_session animal_num
    
    % Initialize.
    selectivity_index_session = [];
    
    for session_num = 1:numel(tuning_task1{animal_num})
        clearvars -except policy tuning_task1 selectivity_index_animal_session animal_num selectivity_index_session session_num
        
        tuning_object_dir = tuning_task1{animal_num}{session_num}.tuning_object_dir;
        object_vel_cell_idx = tuning_task1{animal_num}{session_num}.object_vel_cell_idx;
        
        if ~isempty(object_vel_cell_idx{1}) == 1 && ~isempty(object_vel_cell_idx{2}) == 1
            region_num_temp = 1; region = 2;
        elseif ~isempty(object_vel_cell_idx{1}) == 0 && ~isempty(object_vel_cell_idx{2}) == 1
            region_num_temp = 2; region = 2;
        elseif ~isempty(object_vel_cell_idx{1}) == 1 && ~isempty(object_vel_cell_idx{2}) == 0
            region_num_temp = 1; region = 2;
        end
        
        angles = [0:2*pi/8:2*pi];
        angles = angles(1:end - 1); % Remove the last one.
        
        % Initialize.
        selectivity_index_region = [];
        
        if isempty(object_vel_cell_idx{1}) == 1 && isempty(object_vel_cell_idx{2}) == 1
            selectivity_index_region = [];
        else
            for region_num = region_num_temp:region
                for cell_num = 1:length(object_vel_cell_idx{region_num})
                    response = [];
                    numerator = [];
                    if sum(isnan(tuning_object_dir{region_num}(cell_num,1:8))) == 0 % All 8 directions exist.
                        response = tuning_object_dir{region_num}(object_vel_cell_idx{region_num}(cell_num),1:8);
                    else % All 8 directions do not exist.
                        response = tuning_object_dir{region_num}(object_vel_cell_idx{region_num}(cell_num),1:8);
                        response(find(isnan(tuning_object_dir{region_num}(cell_num,1:8)))) = nanmin(response); % Replace nan with the minimum value.
                    end
                    
                    for dir_bin = 1:8
                        numerator(dir_bin) = response(dir_bin)*exp(1i*2*angles(dir_bin));
                    end
                    
                    cv{region_num}(cell_num) = 1 - abs(sum(numerator)/sum(response));
                    selectivity_index{region_num}(cell_num) = 1 - cv{region_num}(cell_num);
                end
                
                % Concatenate across regions.
                selectivity_index_region = [selectivity_index_region,selectivity_index{region_num}];
            end
        end
        
        % Concatenate across sessions.
        selectivity_index_session = [selectivity_index_session,selectivity_index_region];
    end
    
    % Concatenate across animals.
    selectivity_index_animal_session = [selectivity_index_animal_session,selectivity_index_session];
end

end