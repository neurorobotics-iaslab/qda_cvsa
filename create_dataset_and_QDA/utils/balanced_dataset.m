%% Balance the dataset according to the number of trials
function [X_bal,y_bal,new_trialStart,new_trialDur] = balanced_dataset(dataset, cues, classes, trial_start, trial_dur)

% Check if the dataset is generated with equal number of classes
Ck = cues(trial_start);
class1_idx = find(Ck == classes(1));
class2_idx = find(Ck == classes(2));
class1 = numel(class1_idx);
class2 = numel(class2_idx);

min_num_trials = min(class1, class2);
if class1 > class2
    class1_idx = class1_idx(1:min_num_trials);
else
    class2_idx = class2_idx(1:min_num_trials);
end

sel_trials = sort([class1_idx;class2_idx]);
new_trialStart = trial_start(sel_trials);
new_trialDur = trial_dur(sel_trials);
% create new balanced dataset
X_bal = [];
y_bal = [];
for i = 1:length(sel_trials)
    start_idx = new_trialStart(i);
    dur = new_trialDur(i);
    trial_data = dataset(start_idx:(start_idx + dur),:);
    trial_label = cues(start_idx:(start_idx + dur),:);
    y_bal = cat(1,y_bal,trial_label);
    X_bal = cat(1,X_bal, trial_data);
end
