function get_direction_tuning_selectivity_task1_agent

close all
clear all
clc

% Get direction tuning selectivity.
% Load 'tuning_task1' after running 'get_tuning_curve_task1_agent.m'.

% Initialize.
selectivity_index_agent_temp = [];

for seed_num = 1:numel(tuning_task1)
    clearvars -except policy tuning_task1 selectivity_index_agent_temp seed_num
    
    q_dir = tuning_task1{seed_num}.q_dir;
    
    angles = [0:2*pi/8:2*pi];
    angles = angles(1:end - 1); % Remove the last one.
    
    % Initialize.
    selectivity_index_layer = [];
    
    % Q-network.
    for layer_num = 1:3
        for cell_num = 1:size(q_dir{layer_num},1)
            response = [];
            numerator = [];
            response = q_dir{layer_num}(cell_num,:);
            
            for dir_bin = 1:8
                numerator(dir_bin) = response(dir_bin)*exp(1i*2*angles(dir_bin));
            end
            
            cv{layer_num}(cell_num) = 1 - abs(sum(numerator)/sum(response));
            selectivity_index{layer_num}(cell_num) = 1 - cv{layer_num}(cell_num);
        end
        
        % Concatenate across layers.
        selectivity_index_layer = [selectivity_index_layer,selectivity_index{layer_num}];
    end
    
    % Concatenate across agents.
    selectivity_index_agent_temp = [selectivity_index_agent_temp,selectivity_index_layer];
end

% Remove nans.
selectivity_index_agent = selectivity_index_agent_temp(~isnan(selectivity_index_agent_temp));

end