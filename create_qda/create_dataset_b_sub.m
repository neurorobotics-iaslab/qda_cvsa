% baseline subtraction!
clear all; % close all;

addpath('/home/paolo/bci_vr_ws/src/analysis_bci/equal_ros')
addpath('/home/paolo/bci_vr_ws/src/analysis_bci/utils')

%% Initialization
DATAPAH = '/home/paolo/bci_vr_ws/src/';

% channels_label = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'Fp1', 'CP1', 'CPz', 'CP2', 'Fp2'};


%% Load file
[filenames, pathname] = uigetfile('*.gdf', 'Select GDF Files', 'MultiSelect', 'on');
if ischar(filenames)
    filenames = {filenames};
end
[~, name_no_ext, ~] = fileparts(filenames{1});
parts = strsplit(name_no_ext, '.');
paradigm = parts{end};
subject = filenames{1}(1:2);
time_str = datestr(now, 'ddmmyyyy_HHMMSS');
save_path_qda_dataset = [DATAPAH 'qda_bci/create_qda/datasets/' paradigm '/data.' subject '.' time_str '.' paradigm '.mat'];

%% Initialization
nFiles = length(filenames);
peaks = zeros(1, nFiles);
for idx_file = 1:nFiles
    fullpath_file = fullfile(pathname, filenames{idx_file});
    peaks(idx_file) = analyze_alpha_peak(fullpath_file, 'RestTrigger', 786, 'band', [8 14], 'target_regions', {'C1', 'C3', 'C2', 'C4'});
end

if strcmp(paradigm, 'mi_lhrh')
    % classes = [771 773];
    classes = [769 770];
elseif strcmp(paradigm, 'cvsa_blbr')
    classes = [730 731];
end
nchannels = 32;
nclasses = length(classes);
filterOrder = 4;
avg = 1;
do_hann = false;

%% start processing data
bands = [{[8 14]} {[18 24]}];
bands_str = cellfun(@(x) sprintf('%d-%d', x(1), x(2)), bands, 'UniformOutput', false);
nbands = length(bands);
signals = cell(1, nbands);
artifacts = [];
headers = cell(1, nbands);
for idx_band = 1:nbands
    headers{idx_band}.TYP = [];
    headers{idx_band}.DUR = [];
    headers{idx_band}.POS = [];
    signals{idx_band} = [];
end

for idx_file= 1: nFiles
    fullpath_file = fullfile(pathname, filenames{idx_file});
    disp(['file (' num2str(idx_file) '/' num2str(nFiles)  '): ', filenames{idx_file}]);
    [c_signal, header] = sload(fullpath_file);
    c_signal = c_signal(:,1:nchannels);
    sampleRate = header.SampleRate;

    channels_label = header.Label(1:nchannels);

    excl_ch = {'Fp1', 'Fp2'};
    [found, indices] = ismember(excl_ch, channels_label);
    excl_chs = indices(found);

    % for power band using hilbert transformation and artefact remotion -----------------------------------------------
    bufferSize = floor(avg*sampleRate);
    chunkSize = 25;
    eog.filterOrder = 4;
    eog.band = [1 10];
    eog.label = excl_ch;
    eog.h_threshold = 100;
    eog.v_threshold = 100;
    picks.filterOrder = 4;
    picks.freq = 1; % remove antneuro problems
    picks.threshold = 120;
    artifact = artifact_rejection(c_signal, header, nchannels, bufferSize, chunkSize, eog, picks);
    artifacts = cat(1, artifacts, artifact(:,:));

    disp('   [proc] power band');
    for idx_band = 1:nbands
        band = bands{idx_band};

%         [signal_processed, header_processed] = processing_onlineROS_CAR_hilbert(c_signal, header, nchannels, bufferSize, filterOrder, band, chunkSize, excl_chs, do_hann);
        [signal_processed, header_processed] = processing_onlineROS_hilbert(c_signal, header, nchannels, bufferSize, filterOrder, band, chunkSize, do_hann);
        
        c_header = headers{1, idx_band};
        c_header.sampleRate = header_processed.SampleRate/chunkSize;
        c_header.channels_labels = header_processed.Label;
        if isempty(find(header_processed.EVENT.TYP == 2, 1)) % no eye calibration
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP);
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR);
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS + size(signals{1, idx_band}, 1));
        else
            k = find(header_processed.EVENT.TYP == 1, 1);
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP(k:end));
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR(k:end));
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS(k:end) + size(signals{1, idx_band}, 1));
        end
        signals{1, idx_band} = cat(1, signals{1, idx_band}, signal_processed(:,:));
        headers{1, idx_band} = c_header;
    end
end


%% Labelling data 
events = headers{1,1};
sampleRate = events.sampleRate;
cuePOS = events.POS(ismember(events.TYP, classes));
cueDUR = events.DUR(ismember(events.TYP, classes));
cueTYP = events.TYP(ismember(events.TYP, classes));

fixPOS = events.POS(events.TYP == 786);
fixDUR = events.DUR(events.TYP == 786);

cfPOS = events.POS(events.TYP == 781);
cfDUR = events.DUR(events.TYP == 781);

minDurCue = min(cueDUR);
minDurFix = min(fixDUR);
ntrial = length(cuePOS);

%% Labeling data for the dataset
trial_start = nan(ntrial, 1);
trial_end = nan(ntrial, 1);
trial_typ = nan(ntrial, 1);
for idx_trial = 1:ntrial
    trial_start(idx_trial) = fixPOS(idx_trial);
    trial_typ(idx_trial) = cueTYP(idx_trial);
    trial_end(idx_trial) = cfPOS(idx_trial) + cfDUR(idx_trial) - 1;
end

min_trial_data = min(trial_end - trial_start+1);
trial_data = nan(min_trial_data, nbands, nchannels, ntrial); % data x bands x channels x trial
artifacts_data = nan(min_trial_data, ntrial); % data x trial
for idx_band = 1:nbands
    c_signal = signals{idx_band};
    c_artifact = artifacts;
    for trial = 1:ntrial
        c_start = trial_start(trial);
        c_end = trial_start(trial) + min_trial_data - 1;
        trial_data(:,idx_band,:,trial) = c_signal(c_start:c_end,:);
        artifacts_data(:,trial) = c_artifact(c_start:c_end,:);
    end
end

%% refactoring the data --> odd trial class 1 even class 2
idx_classes_trial = nan(ntrial/2, nclasses);
for idx_class = 1:nclasses
    idx_classes_trial(:,idx_class) = find(trial_typ == classes(idx_class));
end

tmp_data = nan(size(trial_data));
tmp_art = nan(size(artifacts_data));
trial_typ = nan(size(trial_typ));
i = 1;
for idx_trial_class = 1:2:ntrial
    for idx_class = 1:nclasses
        tmp_data(:,:,:,idx_trial_class + idx_class - 1) = trial_data(:,:,:,idx_classes_trial(i, idx_class));
        tmp_art(:,idx_trial_class + idx_class - 1) = artifacts_data(:,idx_classes_trial(i, idx_class));
        trial_typ(idx_trial_class + idx_class - 1) = classes(idx_class);
    end
    i = i + 1;
end
trial_data = tmp_data; % samples x bands x channels x trials
artifacts_data = tmp_art;


%% extract and save data for the QDA
data_cf = trial_data(minDurCue+minDurFix+1:end,:,:,:); % data x bands x channels x trial
data_fix = trial_data(1:minDurFix,:,:,:);
artifacts_cf = artifacts_data(minDurCue+minDurFix+1:end,:,:,:);
artifacts_fix = artifacts_data(1:minDurFix,:,:,:);
nsamples_fix = size(data_fix, 1);
baseline = nan(ntrial, nbands, nchannels);
for idx_band= 1:nbands
    for idx_trial = 1:ntrial
        tmp_X = [];
        for idx_sample = 1:nsamples_fix
            if artifacts_fix(idx_sample, idx_trial) == 0
                tmp_X = [tmp_X; data_fix(idx_sample, idx_band,:,idx_trial)];
            end
        end
        baseline(idx_trial,idx_band,:) = squeeze(mean(tmp_X,1));
    end
    
end
baseline = log(baseline);
nsamples_cf = size(data_cf,1);
X = []; 
y = []; 
count_artifact = 0; count_all = 0;
for idx_band = 1:nbands
    tmp_X = []; 
    y = [];
    trials = []; 
    for idx_trial =  1:ntrial
        for idx_sample = 1:nsamples_cf
            if artifacts_cf(idx_sample,idx_trial) == 0 % no artifact
                c_x = log(data_cf(idx_sample,idx_band,:,idx_trial)) - baseline(idx_trial, idx_band,:);
                tmp_X = [tmp_X; c_x];
                trials = [trials, idx_trial];
                y = [y; trial_typ(idx_trial)];
            else
                count_artifact = count_artifact + 1/nbands;
            end
            count_all = count_all + 1/nbands ;
        end
    end

    X = [X, tmp_X];
end
disp(['all sample for the trials: ' num2str(count_all) ', rejected for artifact: ' num2str(count_artifact)])

%% Features selection QDA
% fisher score
fisher = nan(nbands, nchannels);
label_fisher = [];

for idx_ch=1:nchannels
    for idx_band = 1:nbands
        % all
        mu1 = mean(X(y == classes(1), idx_band, idx_ch));
        sigma1 = std(X(y == classes(1),idx_band, idx_ch));
        mu2 = mean(X(y == classes(2),idx_band, idx_ch));
        sigma2 = std(X(y == classes(2),idx_band, idx_ch));
        fisher(idx_band, idx_ch) = abs(mu1 - mu2)^2 / (sigma1^2 + sigma2^2);
        label_fisher = [label_fisher, {[bands_str{idx_band}]}];
    end
end

figure();
imagesc(fisher')
colorbar;
yticks(1:nchannels); yticklabels(channels_label)
xticks(1:size(fisher, 1)); xticklabels(label_fisher)
sgtitle(['fisher score | ' paradigm])

% R^2
for idx_band = 1:nbands
    r2 = calc_r2_from_data(squeeze(X(:,idx_band,:)), y, 'Plot', true, 'ChanLabels', channels_label, 'title_data', paradigm);
end

%% save data for qda
channels_labels =  [{{'C4', 'C3'}}, {{}}]; % first 8-14 then 18-24
idx_channels = [];
for i = 1:length(channels_labels)
    c_t = channels_labels{i};
    [~, tmp_idx] = ismember(c_t, channels_label);
    idx_channels = [idx_channels, {tmp_idx}];
end
save(save_path_qda_dataset, 'X', 'y', 'trials', 'classes', 'idx_channels', 'channels_labels', 'filenames', 'bands')
disp(['Dataset saved in ', save_path_qda_dataset]);
