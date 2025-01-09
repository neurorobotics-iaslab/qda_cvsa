% create the dataset log band power, selected channels and selected freq
% it works with gdf files of calibration files

% the dataset is created starting form the cue to the end of cf
function [sfile,X,y] = createDataset(path)

%% informations
train_percentage = 0.75;
classes = [730 731];
sampleRate = 512;
filterOrder = 4;

features_file = [path '/dataset/selected_features.mat'];
features = load(features_file);
bands = features.selectedFeatures(:,2);
selchs = features.selectedFeatures(:,1);

sfile = [path '/dataset/dataset_lbp.mat'];
mfile = [path '/mat'];

path = [path '/gdf/calibration'];
files = dir(fullfile(path, '*.gdf'));

channels_label = {'FP1', 'FP2', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'EOG', ...
        'F1', 'F2', 'FC3', 'FCZ', 'FC4', 'C1', 'C2', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};


%% initialization variable to save
X = [];
y = [];
final_bands = [];
for i=1:size(bands,1)
    c_band = bands{i};
    for j=1:size(c_band, 1)
        final_bands = cat(1, final_bands, c_band);
    end
end

info.classes = classes;
info.sampleRate = sampleRate;
info.selchs = selchs;
info.idx_selchs = [];
info.files = {};
info.band = final_bands;
info.trialStart = [];
info.trialDUR = [];
info.filterOrder = 4;
info.startNewFile = [0];
eye_calibration = input("there is eyes calibration? yes (1), no (2): ");


%% take only interested data
for idx_f = 1:length(files)
    file = fullfile(path, files(idx_f).name);
    disp(['file (' num2str(idx_f) '/' num2str(length(files))  '): ', file])
    %load(file);
    [signal, header] = sload(file);
    info.files = cat(1, info.files, files(idx_f).name);
    nchannels = length(channels_label);

    %% labeling
    disp('   Labelling')
    signal = signal(:,1:nchannels);
    events = header.EVENT;
    cuePOS = events.POS(events.TYP == 730 | events.TYP == 731);
    cueDUR = events.DUR(events.TYP == 730 | events.TYP == 731);
    cueTYP = events.TYP(events.TYP == 730 | events.TYP == 731);
    cfPOS  = events.POS(events.TYP == 781);
    cfDUR  = events.DUR(events.TYP == 781);
    nTrials = length(cueTYP);

    trialStart = find(events.TYP == 1);
    targetHit = find(events.TYP == 897 | events.TYP == 898 | events.TYP == 899);
    if(eye_calibration == 1) % ask if eye calibration, if so then there are 2 cue extra
        cuePOS = cuePOS(3:end) - 1;
        cueTYP = cueTYP(3:end);
        cueDUR = cueDUR(3:end);
        nTrials = length(cueTYP);
    end

    %% Initialization variables
    disp('   Initialization variables for the behaviour as rosneuro')
    frameSize = 32;
    bufferSize = 512;
    if idx_f == 1
        prev_file = 0;
    end
    X_band = [];

    for idx_band = 1:length(bands)
        c_band = bands{idx_band};
        disp(['   band: ' num2str(c_band(1)) '-' num2str(c_band(2))]);
        [c_b_low, c_a_low] = butter(filterOrder, c_band(2)*(2/sampleRate),'low');
        [c_b_high, c_a_high] = butter(filterOrder, c_band(1)*(2/sampleRate),'high');
        zi_low = [];
        zi_high = [];
        X_temp = [];
        y_temp = [];

        %% Iterate over trials
        for i=1:nTrials
            disp(['      trial ' num2str(i) '/' num2str(nTrials)])
            % initialization variables
            buffer = nan(bufferSize, nchannels);
            start_trial = cuePOS(i); % in this way the cue is used to fill the buffer
            end_trial = cfPOS(i) + cfDUR(i) - 1;
            % division for frameSize
            end_trial = int64(ceil(single(end_trial-start_trial)/32)*32) + start_trial;
            data = signal(start_trial:end_trial,:);

            % eye movement check
            threshold = 5.5e+04; %[55 mV]
            disp('         Checking data for eye movement')
            result = eye_movement_check(data,channels_label,threshold,sampleRate);
            if result
                disp(['            Eye movement detected: trial ' num2str(i) ' discarded'])
                %skip trial info from header
                events.TYP(trialStart(i):targetHit(i)) = 0;
                events.DUR(trialStart(i):targetHit(i)) = 0;
                events.POS(trialStart(i):targetHit(i)) = 0;

                continue
            else
                disp('            No eye movement detected')
            end

            % application of the buffer
            if idx_band == 1
                info.trialStart = cat(1, info.trialStart, 1+size(X_temp,1)+prev_file);
            end

            nchunks = (end_trial-start_trial) / frameSize;
            for j = 1:nchunks
                frame = data((j-1)*frameSize+1:j*frameSize,:);
                buffer(1:end-frameSize,:) = buffer(frameSize+1:end,:);
                buffer(end-frameSize+1:end, :) = frame;

                % check
                if any(isnan(buffer))
                    continue;
                end

                % apply low and high pass filters
                [s_low, zi_low] = filter(c_b_low,c_a_low,buffer,zi_low);
                [tmp_data,zi_high] = filter(c_b_high,c_a_high,s_low,zi_high);
                %s_band = cat(1, s_band, tmp_data);

                % apply pow
                tmp_data = power(tmp_data, 2);
                %s_pow = cat(1, s_pow, tmp_data);

                % apply average
                tmp_data = mean(tmp_data, 1);
                %s_avg = cat(1, s_avg, tmp_data);

                % apply log
                tmp_data = log(tmp_data);
                %s_log = cat(1, s_log, tmp_data);

                % save in the dataset
                X_temp = cat(1, X_temp, tmp_data);
                y_temp = cat(1, y_temp, repmat(cueTYP(i), size(tmp_data,1), 1));

            end

            % save the dur of the trial only the first time a band is done
            if idx_band == 1
                info.trialDUR = cat(1, info.trialDUR, size(X_temp,1)+prev_file-info.trialStart(end));
            end
        end

        %% take only interested values
        % Check if trials are stored in X_temp
        if isempty(X_temp) 
            disp('All trials skipped')
            continue
        else
            disp('      Take only interested channels for that band')
            % In this way it's possible to keep all the previuosly selected
            % channels from the UI and compare them to the string of known
            % channel labels -> ATTENTION: we repeat the processing for each feature
            selch = selchs(idx_band);
            idx_interest_ch = find(strcmp(channels_label, selch));
    
            % as before
            if idx_band == 1
                info.startNewFile = cat(1, info.startNewFile, size(X,1));
                y = cat(1, y, y_temp);
            end
           
            X_band = cat(2, X_band, X_temp(:,idx_interest_ch));
    
            if idx_f == 1
                info.idx_selchs = cat(2, info.idx_selchs, idx_interest_ch);
            end
        end
    end
   
    X = cat(1, X, X_band);
    prev_file = size(X,1);
    
    % To save a matlab file, with signal and corrected events, for each gdf
    % file after trial removal due to eye movement check
    [~, pfilename] = fileparts(files(idx_f).name);        
    sfilename = [pfilename '.mat'];
    m_path = [mfile '/' sfilename];
    save(m_path, 'header','signal');
end

if ~isempty(info.trialStart)
    info.startTest = info.trialStart(floor(train_percentage * size(info.trialStart,1)));

    % Checks if the dataset is generated with equal number of classes
    n_class1 = sum(y==classes(1));
    n_class2 = sum(y==classes(2));
    if n_class1 ~= n_class2
        disp('Regenerate the dataset with balanced number of trials per class')
        [X_new,y_new,info.trialStart,info.trialDUR] = balanced_dataset(X,y,classes,info.trialStart,info.trialDUR);
        X = X_new;
        y = y_new;
    end
    disp('Save dataset variables.')
    %% save the values
    save(sfile, 'X', 'y', 'info');
else
    disp('No trials to be saved')

end




