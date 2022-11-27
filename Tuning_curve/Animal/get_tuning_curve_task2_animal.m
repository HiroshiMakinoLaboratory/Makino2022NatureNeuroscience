function get_tuning_curve_task2_animal

close all
clear all
clc

% Get tuning curve.

load('animal_activity_task2.mat')

for animal_num = 1:numel(activity_task2)
    clearvars -except activity_task2 animal_num tuning_task2
    
    for session_num = 1:numel(activity_task2{animal_num})
        clearvars -except activity_task2 animal_num session_num tuning_task2
        
        GLM = activity_task2{animal_num}{session_num};
        design_matrix = GLM.design_matrix;
        activity_matrix = GLM.activity_matrix;
        explained_variance_reduction = GLM.explained_variance_reduction;
        p_value_explained_variance = GLM.p_value_explained_variance;
        p_value_variable = GLM.p_value_variable;
        
        % Full model.
        train_frame = GLM.train_frame;
        test_frame = GLM.test_frame;
        B0 = GLM.B0;
        coeff = GLM.coeff;
        predict_train = GLM.predict_train;
        predict_test = GLM.predict_test;
        y_test = GLM.y_test;
        y_hat_test = GLM.y_hat_test;
        y_null_test = GLM.y_null_test;
        L1_test = GLM.L1_test;
        L0_test = GLM.L0_test;
        LS_test = GLM.LS_test;
        explained_variance_test = GLM.explained_variance_test;
        frame_end = GLM.frame_end;
        lick_onset = GLM.lick_onset;
        
        % Predictor index.
        LED_onset_idx = 1:6;
        joystick_vel_idx = 7:54;
        LED_offset_idx = 55:60;
        lick_onset_idx = 61:66;
        reward_onset_idx = 67:75;
        
        % Obtain conditional probability of neural activity.
        if ~isempty(GLM.activity_matrix{1}) == 1 && ~isempty(GLM.activity_matrix{2}) == 1
            region_num_temp = 1; region = 2;
        elseif ~isempty(GLM.activity_matrix{1}) == 0 && ~isempty(GLM.activity_matrix{2}) == 1
            region_num_temp = 2; region = 2;
        elseif ~isempty(GLM.activity_matrix{1}) == 1 && ~isempty(GLM.activity_matrix{2}) == 0
            region_num_temp = 1; region = 2;
        end
        for region_num = region_num_temp:region
            for cell_num = 1:size(GLM.activity_matrix{region_num},1)
                % Scaling factor marginalizing out the effect of the other variables.
                mean_y_hat_LED_onset{region_num}(cell_num) = mean(exp(design_matrix(:,LED_onset_idx)*coeff{region_num}(cell_num,LED_onset_idx)'));
                mean_y_hat_joystick_vel{region_num}(cell_num) = mean(exp(design_matrix(:,joystick_vel_idx)*coeff{region_num}(cell_num,joystick_vel_idx)'));
                mean_y_hat_LED_offset{region_num}(cell_num) = mean(exp(design_matrix(:,LED_offset_idx)*coeff{region_num}(cell_num,LED_offset_idx)'));
                mean_y_hat_lick_onset{region_num}(cell_num) = mean(exp(design_matrix(:,lick_onset_idx)*coeff{region_num}(cell_num,lick_onset_idx)'));
                mean_y_hat_reward_onset{region_num}(cell_num) = mean(exp(design_matrix(:,reward_onset_idx)*coeff{region_num}(cell_num,reward_onset_idx)'));
                
                % Model assessment by measuing explained variance based on Benjamin et al 2018.
                y_test{region_num}(cell_num,:) = activity_matrix{region_num}(cell_num,test_frame);
                y_null_test{region_num}(cell_num) = mean(activity_matrix{region_num}(cell_num,test_frame),2);
                L0_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_null_test{region_num}(cell_num)) - y_null_test{region_num}(cell_num)); % Null model.
                LS_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_test{region_num}(cell_num,:)) - y_test{region_num}(cell_num,:)); % Saturated model.
                
                % LED onset.
                y_hat_LED_onset{region_num}(cell_num,:) = exp(design_matrix(:,LED_onset_idx)*coeff{region_num}(cell_num,LED_onset_idx)').*...
                    mean_y_hat_joystick_vel{region_num}(cell_num).*...
                    mean_y_hat_LED_offset{region_num}(cell_num).*...
                    mean_y_hat_lick_onset{region_num}(cell_num).*...
                    mean_y_hat_reward_onset{region_num}(cell_num).*...
                    exp(B0{region_num}(cell_num));
                y_hat_LED_onset_test{region_num}(cell_num,:) = y_hat_LED_onset{region_num}(cell_num,test_frame);
                L1_LED_onset_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_hat_LED_onset_test{region_num}(cell_num,:)) - y_hat_LED_onset_test{region_num}(cell_num,:)); % Partial model.
                explained_variance_LED_onset_test{region_num}(cell_num) = 1 - (LS_test{region_num}(cell_num) - L1_LED_onset_test{region_num}(cell_num))/(LS_test{region_num}(cell_num) - L0_test{region_num}(cell_num));
                
                % Joystick velocity.
                y_hat_joystick_vel{region_num}(cell_num,:) = exp(design_matrix(:,joystick_vel_idx)*coeff{region_num}(cell_num,joystick_vel_idx)').*...
                    mean_y_hat_LED_onset{region_num}(cell_num).*...
                    mean_y_hat_LED_offset{region_num}(cell_num).*...
                    mean_y_hat_lick_onset{region_num}(cell_num).*...
                    mean_y_hat_reward_onset{region_num}(cell_num).*...
                    exp(B0{region_num}(cell_num));
                y_hat_joystick_vel_test{region_num}(cell_num,:) = y_hat_joystick_vel{region_num}(cell_num,test_frame);
                L1_joystick_vel_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_hat_joystick_vel_test{region_num}(cell_num,:)) - y_hat_joystick_vel_test{region_num}(cell_num,:)); % Partial model.
                explained_variance_joystick_vel_test{region_num}(cell_num) = 1 - (LS_test{region_num}(cell_num) - L1_joystick_vel_test{region_num}(cell_num))/(LS_test{region_num}(cell_num) - L0_test{region_num}(cell_num));
                
                % LED offset.
                y_hat_LED_offset{region_num}(cell_num,:) = exp(design_matrix(:,LED_offset_idx)*coeff{region_num}(cell_num,LED_offset_idx)').*...
                    mean_y_hat_LED_onset{region_num}(cell_num).*...
                    mean_y_hat_joystick_vel{region_num}(cell_num).*...
                    mean_y_hat_lick_onset{region_num}(cell_num).*...
                    mean_y_hat_reward_onset{region_num}(cell_num).*...
                    exp(B0{region_num}(cell_num));
                y_hat_LED_offset_test{region_num}(cell_num,:) = y_hat_LED_offset{region_num}(cell_num,test_frame);
                L1_LED_offset_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_hat_LED_offset_test{region_num}(cell_num,:)) - y_hat_LED_offset_test{region_num}(cell_num,:)); % Partial model.
                explained_variance_LED_offset_test{region_num}(cell_num) = 1 - (LS_test{region_num}(cell_num) - L1_LED_offset_test{region_num}(cell_num))/(LS_test{region_num}(cell_num) - L0_test{region_num}(cell_num));
                
                % Lick onset.
                y_hat_lick_onset{region_num}(cell_num,:) = exp(design_matrix(:,lick_onset_idx)*coeff{region_num}(cell_num,lick_onset_idx)').*...
                    mean_y_hat_LED_onset{region_num}(cell_num).*...
                    mean_y_hat_joystick_vel{region_num}(cell_num).*...
                    mean_y_hat_LED_offset{region_num}(cell_num).*...
                    mean_y_hat_reward_onset{region_num}(cell_num).*...
                    exp(B0{region_num}(cell_num));
                y_hat_lick_onset_test{region_num}(cell_num,:) = y_hat_lick_onset{region_num}(cell_num,test_frame);
                L1_lick_onset_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_hat_lick_onset_test{region_num}(cell_num,:)) - y_hat_lick_onset_test{region_num}(cell_num,:)); % Partial model.
                explained_variance_lick_onset_test{region_num}(cell_num) = 1 - (LS_test{region_num}(cell_num) - L1_lick_onset_test{region_num}(cell_num))/(LS_test{region_num}(cell_num) - L0_test{region_num}(cell_num));
                
                % Reward onset.
                y_hat_reward_onset{region_num}(cell_num,:) = exp(design_matrix(:,reward_onset_idx)*coeff{region_num}(cell_num,reward_onset_idx)').*...
                    mean_y_hat_LED_onset{region_num}(cell_num).*...
                    mean_y_hat_joystick_vel{region_num}(cell_num).*...
                    mean_y_hat_LED_offset{region_num}(cell_num).*...
                    mean_y_hat_lick_onset{region_num}(cell_num).*...
                    exp(B0{region_num}(cell_num));
                y_hat_reward_onset_test{region_num}(cell_num,:) = y_hat_reward_onset{region_num}(cell_num,test_frame);
                L1_reward_onset_test{region_num}(cell_num) = sum(y_test{region_num}(cell_num,:).*log(eps + y_hat_reward_onset_test{region_num}(cell_num,:)) - y_hat_reward_onset_test{region_num}(cell_num,:)); % Partial model.
                explained_variance_reward_onset_test{region_num}(cell_num) = 1 - (LS_test{region_num}(cell_num) - L1_reward_onset_test{region_num}(cell_num))/(LS_test{region_num}(cell_num) - L0_test{region_num}(cell_num));
            end
        end
        
        % Obtain lick tuning.
        lick_onset_frame = find(lick_onset(1:frame_end) == 1);
        lick_onset_frame_all_temp = [];
        for lick_onset_num = 1:length(lick_onset_frame)
            lick_onset_frame_all_temp = [lick_onset_frame_all_temp,(lick_onset_frame(lick_onset_num) - 3):(lick_onset_frame(lick_onset_num) + 3)]; % 1 second bin.
        end
        lick_onset_frame_all = unique(lick_onset_frame_all_temp);
        lick_onset_frame_all = lick_onset_frame_all(lick_onset_frame_all <= length(lick_onset(1:frame_end)));
        all_lick_onset_frame = [1:length(lick_onset(1:frame_end))];
        lick_onset_off_frame_all = all_lick_onset_frame(~ismember(all_lick_onset_frame,lick_onset_frame_all));
        
        for region_num = region_num_temp:region
            for cell_num = 1:size(GLM.activity_matrix{region_num},1)
                tuning_lick_onset{region_num}(cell_num,1) = mean(y_hat_lick_onset{region_num}(cell_num,lick_onset_off_frame_all));
                tuning_lick_onset{region_num}(cell_num,2) = mean(y_hat_lick_onset{region_num}(cell_num,lick_onset_frame_all));
            end
        end
        
        % Choose cells whose explained variance is statistically significant with Benjamini-Hochberg false discovery rate.
        alpha = 0.05;
        % Task-related cells.
        p_value_both = [p_value_explained_variance{1},p_value_explained_variance{2}];
        [sorted_p_value_both,sorted_p_value_both_idx] = sort(p_value_both);
        p_value_both_rank = [1:length(p_value_both)];
        adjusted_p_value = (p_value_both_rank./length(p_value_both)).*alpha;
        adjusted_p_value_sig = sorted_p_value_both < adjusted_p_value;
        sorted_p_value_both_idx_sig = [sorted_p_value_both_idx',adjusted_p_value_sig'];
        unsorted_p_value_both_idx_sig = sortrows(sorted_p_value_both_idx_sig,1);
        unsorted_adjusted_p_value_sig = unsorted_p_value_both_idx_sig(:,2);
        sig_cell_idx{1} = find(unsorted_adjusted_p_value_sig(1:length(p_value_explained_variance{1})))';
        sig_cell_idx{2} = find(unsorted_adjusted_p_value_sig((length(p_value_explained_variance{1}) + 1):end))';
        
        % Variable-related cells.
        p_value_variables_both = [p_value_variable{1};p_value_variable{2}];
        p_value_variables_both_all = p_value_variables_both(:)';
        [sorted_p_value_variables_both_all,sorted_p_value_variables_both_all_idx] = sort(p_value_variables_both_all);
        p_value_variables_both_all_rank = [1:length(p_value_variables_both_all)];
        adjusted_p_value_variables = (p_value_variables_both_all_rank./length(p_value_variables_both_all)).*alpha;
        adjusted_p_value_variables_sig = sorted_p_value_variables_both_all < adjusted_p_value_variables;
        sorted_p_value_variables_both_all_idx_sig = [sorted_p_value_variables_both_all_idx',adjusted_p_value_variables_sig'];
        unsorted_p_value_variables_both_all_idx_sig = sortrows(sorted_p_value_variables_both_all_idx_sig,1);
        unsorted_adjusted_p_value_variables_sig = unsorted_p_value_variables_both_all_idx_sig(:,2);
        reshaped_unsorted_adjusted_p_value_variables_sig = reshape(unsorted_adjusted_p_value_variables_sig,[size(p_value_variables_both,1),size(p_value_variables_both,2)]);
        contribution_matrix_temp1{1} = reshaped_unsorted_adjusted_p_value_variables_sig((1:size(p_value_variable{1},1)),:);
        contribution_matrix_temp1{2} = reshaped_unsorted_adjusted_p_value_variables_sig((size(p_value_variable{1},1) + 1):end,:);
        
        % Get variable-contributing cell indices.
        for region_num = region_num_temp:region
            % Choose cells whose explained variance is statistically significant and bigger than 0.0001.
            explained_variance_pos_cell_idx{region_num} = find(GLM.explained_variance_test{region_num} > 0.0001);
            sig_cell{region_num} = sig_cell_idx{region_num}(ismember(sig_cell_idx{region_num},explained_variance_pos_cell_idx{region_num}));
            
            % Choose cells whose explained variance for each variable is bigger than 0.0001.
            LED_onset_cell_idx_temp1{region_num} = find(explained_variance_LED_onset_test{region_num} > 0.0001);
            joystick_vel_cell_idx_temp1{region_num} = find(explained_variance_joystick_vel_test{region_num} > 0.0001);
            LED_offset_cell_idx_temp1{region_num} = find(explained_variance_LED_offset_test{region_num} > 0.0001);
            lick_onset_cell_idx_temp1{region_num} = find(explained_variance_lick_onset_test{region_num} > 0.0001);
            reward_onset_cell_idx_temp1{region_num} = find(explained_variance_reward_onset_test{region_num} > 0.0001);
            
            % Get a contribution matrix for variables.
            contribution_matrix_temp2{region_num} = explained_variance_reduction{region_num} < 0; % Obtain a binary matrix in which 1 is assigned when explained variance is reduced by removing a variable of interest.
            contribution_matrix{region_num} = contribution_matrix_temp1{region_num} + contribution_matrix_temp2{region_num} == 2; % Obtain a binary matrix in which 1 is assigned when both conditions are met.
            
            % Choose cells according to the contribution matrix.
            LED_onset_cell_idx_temp2{region_num} = find(contribution_matrix{region_num}(:,1));
            joystick_vel_cell_idx_temp2{region_num} = find(contribution_matrix{region_num}(:,2));
            LED_offset_cell_idx_temp2{region_num} = find(contribution_matrix{region_num}(:,3));
            lick_onset_cell_idx_temp2{region_num} = find(contribution_matrix{region_num}(:,4));
            reward_onset_cell_idx_temp2{region_num} = find(contribution_matrix{region_num}(:,5));
            
            % Choose cells for which these conditions are met.
            LED_onset_cell_idx_temp{region_num} = LED_onset_cell_idx_temp1{region_num}(ismember(LED_onset_cell_idx_temp1{region_num},LED_onset_cell_idx_temp2{region_num}));
            joystick_vel_cell_idx_temp{region_num} = joystick_vel_cell_idx_temp1{region_num}(ismember(joystick_vel_cell_idx_temp1{region_num},joystick_vel_cell_idx_temp2{region_num}));
            LED_offset_cell_idx_temp{region_num} = LED_offset_cell_idx_temp1{region_num}(ismember(LED_offset_cell_idx_temp1{region_num},LED_offset_cell_idx_temp2{region_num}));
            lick_onset_cell_idx_temp{region_num} = lick_onset_cell_idx_temp1{region_num}(ismember(lick_onset_cell_idx_temp1{region_num},lick_onset_cell_idx_temp2{region_num}));
            reward_onset_cell_idx_temp{region_num} = reward_onset_cell_idx_temp1{region_num}(ismember(reward_onset_cell_idx_temp1{region_num},reward_onset_cell_idx_temp2{region_num}));
            
            % Choose cells for which all of the above conditions are met.
            LED_onset_cell_idx{region_num} = sig_cell{region_num}(ismember(sig_cell{region_num},LED_onset_cell_idx_temp{region_num}));
            joystick_vel_cell_idx{region_num} = sig_cell{region_num}(ismember(sig_cell{region_num},joystick_vel_cell_idx_temp{region_num}));
            LED_offset_cell_idx{region_num} = sig_cell{region_num}(ismember(sig_cell{region_num},LED_offset_cell_idx_temp{region_num}));
            lick_onset_cell_idx{region_num} = sig_cell{region_num}(ismember(sig_cell{region_num},lick_onset_cell_idx_temp{region_num}));
            reward_onset_cell_idx{region_num} = sig_cell{region_num}(ismember(sig_cell{region_num},reward_onset_cell_idx_temp{region_num}));
        end
        
        tuning_animal.tuning_lick_onset = tuning_lick_onset;
        tuning_animal.LED_onset_cell_idx = LED_onset_cell_idx;
        tuning_animal.joystick_vel_cell_idx = joystick_vel_cell_idx;
        tuning_animal.lick_onset_cell_idx = lick_onset_cell_idx;
        tuning_animal.reward_onset_cell_idx = reward_onset_cell_idx;
        
        tuning_task2{animal_num}{session_num} = tuning_animal;
    end
end

end